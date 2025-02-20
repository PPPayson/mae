import math
import copy
import timm
import os
import json
import numpy as np
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tvF
from matplotlib import pyplot as plt
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.models.models_encoder import LinearModel, FullModel, LinearModelTimm
from src.data.co3d_dataset import EmbeddingDataset
from src.data.datasets import build_co3d_eval_loader
from src.util.save_features import extract_features
from PIL import Image
import src.util.clickme_utils as clickme_utils
import src.util.misc as misc
import src.util.metrics as metrics


def dino_eval_one_epoch(model: torch.nn.Module, 
                        model_without_ddp: torch.nn.Module,
                        data_loader: Iterable,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        log_writer=None, 
                        start_steps=0,
                        camera_params_enabled=False,
                        schedule_free=False, 
                        categorical_camera=False, 
                        alpha=0.5,
                        args=None):
    model.eval()
    if schedule_free:
        optimizer.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval Epoch: [{}]'.format(epoch)
    print_freq = 10
    params = model.parameters()
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with torch.no_grad():
            videos = batch[0]
            true_camera_params = None
            if camera_params_enabled:
                true_camera_params = batch[2].to(device, non_blocking=True)
                camera_param_cats = batch[3].to(device, non_blocking=True)
            videos = videos.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(videos)
                loss = model_without_ddp.calculate_loss(outputs['student_output'], outputs['teacher_output'], epoch)
                loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        metric_logger.synchronize_between_processes()
    if log_writer is not None:
        log_writer.set_step(start_steps)
        log_writer.update(loss=metric_logger.loss.global_avg, head="val")
        log_writer.update(epoch=epoch, head="val")
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def autoreg_eval_one_epoch(model: torch.nn.Module,
                        data_loader: Iterable, 
                        device: torch.device, 
                        epoch: int, 
                        patch_size: int = 16,
                        normalize_target: bool = True, 
                        log_writer=None, 
                        start_steps=None, 
                        use_cce=True, 
                        n_frames=4,
                        camera_params_enabled=False, 
                        categorical_camera=False, 
                        alpha=(0.5, 0.5),
                        linear_model=None,
                        feature_loss=False, 
                        args=None):
    model.eval()
    if linear_model is not None:
        linear_model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'EVAL epoch: [{}]'.format(epoch)
    print_freq = 10
    mean = torch.as_tensor(args.mean).to(device)[None, :, None, None, None]
    std = torch.as_tensor(args.std).to(device)[None, :, None, None, None]
    if use_cce:
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.MSELoss()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with torch.no_grad():
            videos, bool_masked_pos = batch[0], batch[1]
            true_camera_params = None
            if camera_params_enabled:
                true_camera_params = batch[2].to(device, non_blocking=True)
                camera_param_cats = batch[3].to(device, non_blocking=True)
                if feature_loss:
                    features = batch[4].to(device, non_blocking=True)
            elif feature_loss:
                features = batch[2].to(device, non_blocking=True)
            videos = videos.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            unnorm_videos = videos * std + mean  # in [0, 1]
            videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=1, p1=patch_size, p2=patch_size)
            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(videos, camera=true_camera_params, mask=bool_masked_pos)
                pred_frames = outputs['pred_frames']
                if use_cce:
                    labels_long = (labels*15).to(torch.long)
                    if args.mask_type == 'autoregressive':
                        pred_frames = rearrange(pred_frames, 'b (t n) (p c) -> b t c n p', c=16, t=n_frames-1)
                        pred_frames = pred_frames[:, 1:, :, :, :]
                        pred_frames = rearrange(pred_frames, 'b t c n p -> b c (t n) p', c=16, t=n_frames-2)
                    else:
                        pred_frames = rearrange(pred_frames, 'b n (p c) -> b c n p', c=16)
                    loss = loss_func(input=pred_frames, target=labels_long)

                else:
                    if args.mask_type == 'autoregressive':
                        pred_frames = rearrange(pred_frames, 'b (t n) p -> b t n p', t=n_frames-1)
                        pred_frames = pred_frames[:, 1:, :, :]
                        pred_frames = rearrange(pred_frames, 'b t n p -> b (t n) p', t=n_frames-2)                                
                    loss = loss_func(input=pred_frames, target=labels)
                if use_cce:
                    loss = loss/math.log(16)
                recon_loss_value = loss.item()
                metric_logger.update(recon_loss=recon_loss_value)
                if camera_params_enabled:
                    pred_camera = outputs['pred_camera']
                    gt_camera = outputs['gt_camera']
                    if categorical_camera:
                        camera_param_cats = camera_param_cats[:, 2:]
                        pred_camera = rearrange(pred_camera, 'b t c -> b c t')[:, :, 1:]
                        cam_loss = F.cross_entropy(input=pred_camera, target=camera_param_cats)/math.log(64)
                    else:
                        cam_loss = F.mse_loss(input=pred_camera, target=gt_embed)
                    cam_loss_value = cam_loss.item()
                    loss += alpha[0]*cam_loss
                    metric_logger.update(cam_loss=cam_loss_value)
                if feature_loss and linear_model is not None:
                    new_features = linear_model(outputs['encoder_features'])
                    if args.mask_type == 'causal' or args.mask_type=="autoregressive":
                        features = rearrange(features[:, :n_frames-1, :], 'b t c -> (b t) c')
                    else:
                        features = rearrange(features, 'b t c -> (b t) c')
                    features = F.softmax(features)
                    f_loss = F.cross_entropy(new_features, features)/math.log(1000)
                    f_loss_value = f_loss.item()
                    loss += alpha[1]*f_loss
                    metric_logger.update(feature_loss=f_loss_value)
                loss_value = loss.item()
                metric_logger.update(loss=loss_value)
                metric_logger.synchronize_between_processes()
    if log_writer is not None:
        log_writer.set_step(start_steps)
        log_writer.update(loss=metric_logger.loss.global_avg, head="val")
        log_writer.update(epoch=epoch, head="val")
        log_writer.update(recon_loss=metric_logger.recon_loss.global_avg, head="val")
        if camera_params_enabled:
            log_writer.update(cam_loss=metric_logger.cam_loss.global_avg, head='val')
        if feature_loss and linear_model is not None:
            log_writer.update(feature_loss=metric_logger.feature_loss.global_avg, head='val')
        reconstruction = videos_patch.clone()
        if use_cce:
            pred_frames = torch.argmax(pred_frames, dim=1)/15.0
        reconstruction[bool_masked_pos] = pred_frames.view(B*pred_frames.shape[1], -1).float()
        reconstruction = rearrange(reconstruction, 'b n (p c) -> b n p c', c=3)
        videos_patch = rearrange(unnorm_videos[0:1], 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=1, p1=patch_size, p2=patch_size)
        videos_patch_to_mask = copy.deepcopy(videos_patch)
        videos_patch_to_mask[bool_masked_pos[0:1]] = 0
        masked_inp = rearrange(videos_patch_to_mask, 'b (t h w) (p0 p1 p2 c) -> b (t p0) c (h p1) (w p2)', p0=1, p1=patch_size, p2=patch_size, w=14, h=14, t=n_frames)
        reconstruction = rearrange(reconstruction, 'b (t h w) (p0 p1 p2) c -> b (t p0) c (h p1) (w p2)', p0=1, p1=patch_size, p2=patch_size, w=14, h=14, t=n_frames)
        if use_cce:
            unnorm_videos = (unnorm_videos*15).to(torch.long)/15.
        log_writer.update(valframes=[unnorm_videos[0].transpose(0, 1), reconstruction[0], masked_inp[0]], head="val")
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, 
                    epoch: int, log_writer=None, start_steps=0,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'EVAL epoch: [{}]'.format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter
    mean = torch.as_tensor(args.mean).to(device)[None, :, None, None]
    std = torch.as_tensor(args.std).to(device)[None, :, None, None]

    with torch.no_grad():
        for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples[0]
            if args.video_dataset:
                samples = rearrange(samples, 'b c t h w -> (b t) c h w')

            samples = samples.to(device, non_blocking=True)

            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss, pred, _ = model(samples, mask_ratio=args.mask_ratio)
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
    if log_writer is not None:
        log_writer.set_step(start_steps)
        reconstruction = pred
        log_writer.update(loss=metric_logger.loss.avg, head="val")
        reconstruction = rearrange(reconstruction, 'b n (p c) -> b n p c', c=3)
        reconstruction = rearrange(reconstruction, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=16, p2=16, w=14, h=14)
        reconstruction = reconstruction * std + mean        
        log_writer.update(valframes=[reconstruction[0]], head="val")

def eval_alignment(full_model:torch.nn.Module, test_data_loader: Iterable, 
                device: torch.device, log_writer, visualize, args,
                kernel_size=21, kernel_sigma=21, save_img=False, dataset='co3d'):

    full_model.eval()
    sample_maps = []

    # ceiling = test_data_loader.dataset.ceiling
    # print("Ceiling: ", ceiling)
    alignment_scores = {'full_spearman':[], 'unnorm_spearman': [], 'ceiling': [], 'topk_spearman': [], 'unnorm_topk': []}
    null_scores = {'full_spearman':[], 'unnorm_spearman': [], 'topk_spearman': [], 'unnorm_topk': []}
    all_test_acc = []

    kernel = clickme_utils.circle_kernel(kernel_size, kernel_sigma).to(device)

    # Evaluate each image
    for i, batch in enumerate(test_data_loader):
        imgs, hmps, labels, img_names, cat, ceiling = batch
        # ceiling = test_data_loader.dataset.ceiling

        imgs, hmps, labels = imgs.to(device), hmps.to(device), labels.to(device)
        img_name = img_names[0]
        cat = cat[0]

        # Select a random hmp for null score
        sub_vec = np.where(np.array(test_data_loader.dataset.categories) != cat)[0]
        random_idx = np.random.choice(sub_vec)
        random_hmps = test_data_loader.dataset[random_idx]
        random_hmps = torch.unsqueeze(torch.Tensor(random_hmps[1]), 0)

        # Get accuracy and loss
        imgs.requires_grad = True
        outputs = full_model(imgs)
        test_acc = misc.accuracy(outputs, labels)[0].item()
        all_test_acc.append(test_acc)
        most_probable_class = torch.argmax(outputs, dim=-1)

        most_probable_scores = outputs[torch.arange(outputs.size(0)), most_probable_class]  # Shape: (batch_size,)
        saliency = torch.autograd.grad(outputs=most_probable_scores, inputs=imgs, 
                                            grad_outputs=torch.ones_like(most_probable_scores),
                                            create_graph=False)[0]
                                            
        saliency = torch.amax(saliency.abs(), dim=1).detach().cpu()

        if saliency.shape[-1] != 224:
            saliency = F.interpolate(saliency.unsqueeze(0), size=(224, 224), mode="bilinear").to(torch.float32)
        # Get average hmps and average half hmps
        hmps = tvF.resize(hmps, 256)
        hmps = tvF.center_crop(hmps, (224, 224))
        hmps = hmps.mean(1)
        random_hmps = tvF.resize(random_hmps, 256)
        random_hmps = tvF.center_crop(random_hmps, (224, 224))
        random_hmps = random_hmps.mean(1)

        hmps = (hmps - hmps.min()) / (hmps.max() - hmps.min())
        random_hmps = (random_hmps - random_hmps.min()) / (random_hmps.max() - random_hmps.min())

        # Get topk model saliency points and half top k to match sparisty of hmps and half hmps
        full_saliency = saliency
        topk_saliency = saliency.clone()
        full_saliency = clickme_utils.gaussian_blur(full_saliency.to(device).unsqueeze(0), kernel)
        # Double convolve
        full_saliency = clickme_utils.gaussian_blur(full_saliency, kernel).squeeze()

        k = torch.sum(hmps>0)
        flat_saliency = topk_saliency.flatten()
        topk, indices = torch.topk(flat_saliency, k)
        thresh_value = topk[-1]
        topk_saliency[topk_saliency<thresh_value] = 0

        topk_saliency = clickme_utils.gaussian_blur(topk_saliency.to(device).unsqueeze(0), kernel)
        topk_saliency = clickme_utils.gaussian_blur(topk_saliency, kernel).squeeze()


        full_saliency = full_saliency.detach().cpu().numpy()
        topk_saliency = topk_saliency.detach().cpu().numpy()

        hmps = hmps.detach().cpu().numpy()
        random_hmps = random_hmps.detach().cpu().numpy()

        # Normalize
        full_saliency = (full_saliency - full_saliency.min())/(full_saliency.max() - full_saliency.min())
        topk_saliency = (topk_saliency - topk_saliency.min())/(topk_saliency.max() - topk_saliency.min())

        # Compute spearman
        full_spearman = clickme_utils.compute_spearman_correlation(full_saliency, hmps)
        full_null_spearman = clickme_utils.compute_spearman_correlation(full_saliency, random_hmps)
        alignment_scores['unnorm_spearman'].append(full_spearman)
        null_scores['unnorm_spearman'].append(full_null_spearman)


        topk_spearman = clickme_utils.compute_spearman_correlation(topk_saliency, hmps)
        topk_null_spearman = clickme_utils.compute_spearman_correlation(topk_saliency, random_hmps)
        alignment_scores['unnorm_topk'].append(topk_spearman)
        null_scores['unnorm_topk'].append(topk_null_spearman)
        alignment_scores['ceiling'].append(ceiling)

        full_spearman /= ceiling
        full_null_spearman /= ceiling
        topk_spearman /= ceiling
        topk_null_spearman /= ceiling

        alignment_scores['topk_spearman'].append(topk_spearman)
        null_scores['topk_spearman'].append(topk_null_spearman)
        alignment_scores['full_spearman'].append(full_spearman)
        null_scores['full_spearman'].append(full_null_spearman)
        # Save image for wandb log
        #if len(visualize)>0 and i in visualize:
        if len(visualize)>0:
            # Save a copy of the img for visualization
            img = imgs.clone().detach().cpu().numpy().squeeze()
            img = np.moveaxis(img, 0, -1)
            img = img*args.std + args.mean
            img = np.uint8(255*img)
            hmps_img = hmps.squeeze()
            f = plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(full_saliency.squeeze())
            plt.axis("off")
            plt.subplot(1, 4, 2)
            plt.imshow(topk_saliency.squeeze())
            plt.axis("off")
            plt.subplot(1, 4, 3)
            hmps_img = (hmps_img - np.min(hmps_img))/np.max(hmps_img)
            plt.imshow(hmps_img)
            plt.axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow(img)
            plt.axis("off")

            f.tight_layout(pad=0)
            f.canvas.draw()
          
            buf = f.canvas.buffer_rgba()
            ncols, nrows = f.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)

            image = torch.unsqueeze(torch.Tensor(image), 0)
            image = image[:, int(math.floor(image.shape[1]/4)):int(image.shape[1] - math.floor(image.shape[1]/4)), :, :]

            if save_img:
                v = torch.moveaxis(image, 3, 1)
                img_grid = make_grid(v, nrow=1, normalize=True, scale_each=False)
                img_grid = tvF.to_pil_image(img_grid.clip(0, 0.996))
                output_dir = os.path.join(args.output_dir, dataset)
                os.makedirs(output_dir, exist_ok=True)
                img_grid.save(os.path.join(output_dir, str(full_spearman.numpy()[0]) + '_' + img_name.split('/')[-1].replace('JPEG', 'png')))
                np.save(os.path.join(output_dir, 'full_' + img_name.split('/')[-1].replace('png', 'npy')), full_saliency)
                np.save(os.path.join(output_dir, 'topk_' + img_name.split('/')[-1].replace('png', 'npy')), topk_saliency)

            if i in visualize:
                print(img_name, full_spearman.numpy()[0], ceiling.numpy()[0], alignment_scores['unnorm_spearman'][-1])
                sample_maps.append(image)
            plt.close()

    avg_test_acc = sum(all_test_acc)/float(len(all_test_acc))
    if log_writer is not None:
        log_writer.update(heatmaps=sample_maps, head=f"{test_data_loader.dataset.dataset_name}_eval")

    return avg_test_acc, alignment_scores, null_scores


def eval_co3d(model: torch.nn.Module, 
            train_data_loader: Iterable, 
            val_data_loader: Iterable, 
            test_data_loader: Iterable, 
            imgnet_loader: Iterable, 
            device: torch.device, 
            epoch: int, 
            num_epochs: int, 
            batch_size: int, 
            learning_rate=5e-4, 
            log_writer=None, 
            start_steps=None, 
            num_workers=16, 
            args=None, 
            eval_align=True,
            linear_probe=True,
            timm_model=False,
            save_img=False):
    model.eval()
    if linear_probe:
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            timm_model = False
        elif hasattr(model, 'teacher'):
            encoder = model.teacher_encoder
            timm_model = False
        elif type(model).__name__ == 'MaskedAutoencoderViT':
            timm_model=False
            encoder=model
        else:
            timm_model=True
            encoder = model
        print("Is timm model", timm_model)
        train_features, train_labels = extract_features(encoder, train_data_loader, device, pool=args.timm_pool, timm_model=timm_model)
        val_features, val_labels = extract_features(encoder, val_data_loader, device, pool=args.timm_pool, timm_model=timm_model)
            
        metric_logger = misc.MetricLogger(delimiter="   ")
        header = f'Co3D EVAL'
        print_freq = 1
        
        train_dataset = EmbeddingDataset(train_features, train_labels)
        val_dataset = EmbeddingDataset(val_features, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if timm_model:
            linear_model = LinearModelTimm(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy())), dropout_rate=0.5).to(device)
        else:
            linear_model = LinearModel(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy())), dropout_rate=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(linear_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        best_acc = 0
        best_model_state = linear_model.state_dict().copy()

        for e in metric_logger.log_every(range(num_epochs), print_freq, header):
            metric_logger.update(epoch=e)
            linear_model.train()
            train_loss = 0
            for batch in train_loader:
                embeddings, labels = batch
                embeddings = embeddings.to(device)
                labels = labels.to(device).type(torch.long)
                preds = linear_model(embeddings)
                loss = criterion(preds, labels)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_train_loss = train_loss / len(train_loader)

            linear_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    embeddings, labels = batch
                    embeddings = embeddings.to(device)
                    labels = labels.to(device)
                    preds = linear_model(embeddings)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(preds.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss / len(val_loader)
            acc = correct/total
            if acc > best_acc:
                best_acc = acc
                best_model_state = linear_model.state_dict().copy()
    
            metric_logger.update(train_loss=avg_train_loss)
            metric_logger.update(val_loss=avg_val_loss)
            metric_logger.update(acc=acc)
            metric_logger.synchronize_between_processes()

        if log_writer is not None:
            log_writer.update(val_acc=best_acc, head='co3d_eval')
            log_writer.update(epoch=epoch, head='co3d_eval')
        linear_model.load_state_dict(best_model_state)
        full_model = FullModel(encoder, linear_model, pool=args.timm_pool, timm_model=timm_model)
    else:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_data_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct/total
        full_model = model
        if log_writer is not None:
                log_writer.update(val_acc=acc, head='co3d_eval')
        best_acc=acc
    if eval_align:
        avg_test_acc, alignment_scores, null_scores = eval_alignment(full_model, test_data_loader, device, log_writer, list(range(10)), args, dataset='co3d', save_img=save_img)
        imgnet_acc, imgnet_align, imgnet_null = eval_alignment(full_model, imgnet_loader, device, log_writer, list(range(10)), args, dataset='imgnet', save_img=save_img)
        print(f"Accuracy: {best_acc} Co3D Alignment: {np.mean(alignment_scores['full_spearman'])} ImgNet Alignment: {np.mean(imgnet_align['full_spearman'])}")
        print(f"Co3D Unnorm: {np.mean(alignment_scores['unnorm_spearman'])} ImgNet Unnorm: {np.mean(imgnet_align['unnorm_spearman'])}")
        print(f"Co3D Topk: {np.mean(alignment_scores['topk_spearman'])} ImgNet TopK: {np.mean(imgnet_align['topk_spearman'])}")
        if log_writer is not None:
            log_writer.update(test_acc=avg_test_acc, head='co3d_eval')
            log_writer.update(full_spearman=np.mean(alignment_scores['full_spearman']), head='co3d_eval')
            log_writer.update(full_unnorm=np.mean(alignment_scores['unnorm_spearman']), head='co3d_eval')
            log_writer.update(full_null=np.mean(null_scores['full_spearman']), head='co3d_eval')
            log_writer.update(full_null_unnorm=np.mean(null_scores['unnorm_spearman']), head='co3d_eval')

            log_writer.update(topk_spearman=np.mean(alignment_scores['topk_spearman']), head='co3d_eval')
            log_writer.update(topk_unnorm=np.mean(alignment_scores['unnorm_topk']), head='co3d_eval')
            log_writer.update(topk_null=np.mean(null_scores['topk_spearman']), head='co3d_eval')
            log_writer.update(topk_null_unnorm=np.mean(null_scores['unnorm_topk']), head='co3d_eval')


            log_writer.update(test_acc=imgnet_acc, head='imgnet_eval')
            log_writer.update(full_spearman=np.mean(imgnet_align['full_spearman']), head='imgnet_eval')
            log_writer.update(full_unnorm=np.mean(imgnet_align['unnorm_spearman']), head='imgnet_eval')
            log_writer.update(full_null=np.mean(imgnet_null['full_spearman']), head='imgnet_eval')
            log_writer.update(full_null_unnorm=np.mean(imgnet_null['unnorm_spearman']), head='imgnet_eval')

            log_writer.update(topk_spearman=np.mean(imgnet_align['topk_spearman']), head='imgnet_eval')
            log_writer.update(topk_unnorm=np.mean(imgnet_align['unnorm_topk']), head='imgnet_eval')
            log_writer.update(topk_null=np.mean(imgnet_null['topk_spearman']), head='imgnet_eval')
            log_writer.update(topk_null_unnorm=np.mean(imgnet_null['unnorm_topk']), head='imgnet_eval')

        return {'acc': best_acc, 'full_co3d_alignment': np.mean(alignment_scores['full_spearman']), 'full_imgnet_alignment': np.mean(imgnet_align['full_spearman']),
                'topk_co3d_alignment': np.mean(alignment_scores['topk_spearman']), 'topk_imgnet_alignment': np.mean(imgnet_align['topk_spearman']), 
                'full_co3d_null': np.mean(null_scores['full_spearman']), 'full_imgnet_null': np.mean(imgnet_null['full_spearman']),
                'topk_co3d_null': np.mean(null_scores['topk_spearman']), 'topk_imgnet_null': np.mean(imgnet_null['topk_spearman']),
                'full_unnorm_co3d_alignment': np.mean(alignment_scores['unnorm_spearman']), 'full_unnorm_imgnet_alignment': np.mean(imgnet_align['unnorm_spearman']),
                'topk_unnorm_co3d_alignment': np.mean(alignment_scores['unnorm_topk']), 'topk_unnorm_imgnet_alignment': np.mean(imgnet_align['unnorm_topk']), 
                'full_unnorm_co3d_null': np.mean(null_scores['unnorm_spearman']), 'full_unnorm_imgnet_null': np.mean(imgnet_null['unnorm_spearman']),
                'topk_unnorm_co3d_null': np.mean(null_scores['unnorm_topk']), 'topk_unnorm_imgnet_null': np.mean(imgnet_null['unnorm_topk']),
                'co3d_ceiling': np.mean(alignment_scores['ceiling']), 'imgnet_ceiling': np.mean(imgnet_align['ceiling'])}
    return best_acc


def eval_vpt(model: torch.nn.Module,
            train_data_loader: Iterable,
            val_data_loader: Iterable,
            test_data_loader: Iterable,
            human_data_loader: Iterable,
            device: torch.device,
            epoch: int,
            num_epochs: int = 50,
            batch_size: int = 128,
            learning_rate = 5e-4,
            log_writer = None,
            start_steps=None,
            num_workers=16,
            args=None,
            timm_model=False):
    model.eval()
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        timm_model = False
    elif hasattr(model, 'teacher'):
        encoder = model.teacher_encoder
        timm_model = False
    elif type(model).__name__ == 'MaskedAutoencoderViT':
        timm_model=False
        encoder=model
    else:
        timm_model=True
        encoder = model    
    train_features, train_labels = extract_features(encoder, train_data_loader, device, pool=args.timm_pool, timm_model=timm_model)
    val_features, val_labels = extract_features(encoder, val_data_loader, device, pool=args.timm_pool, timm_model=timm_model)
    test_features, test_labels = extract_features(encoder, test_data_loader, device, pool=args.timm_pool, timm_model=timm_model)
    human_features, human_labels = extract_features(encoder, human_data_loader, device, pool=args.timm_pool, timm_model=timm_model)

    
    metric_logger = misc.MetricLogger(delimiter="   ")
    header = f'VPT EVAL'
    print_freq = 1
    train_dataset = EmbeddingDataset(train_features, train_labels)
    val_dataset = EmbeddingDataset(val_features, val_labels)
    test_dataset = EmbeddingDataset(test_features, test_labels)
    human_dataset = EmbeddingDataset(human_features, human_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    human_loader = DataLoader(human_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if timm_model:
        linear_model = LinearModelTimm(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy())), dropout_rate=0.5).to(device)
    else:
        linear_model = LinearModel(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy())), dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_train_acc = 0
    best_val_acc = 0
    best_model_state = linear_model.state_dict().copy()

    for e in metric_logger.log_every(range(num_epochs), print_freq, header):
        metric_logger.update(epoch=e)
        linear_model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device).type(torch.long)
            preds = linear_model(embeddings)
            loss = criterion(preds, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct/total
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        avg_train_loss = train_loss / len(train_loader)

        linear_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings, labels = batch
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                preds = linear_model(embeddings)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct/total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = linear_model.state_dict().copy()

        metric_logger.update(train_loss=avg_train_loss)
        metric_logger.update(val_loss=avg_val_loss)
        metric_logger.update(val_acc=val_acc)
        metric_logger.synchronize_between_processes()
    linear_model.load_state_dict(best_model_state)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            preds = linear_model(embeddings)
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        best_test_acc = correct/total

        correct = 0
        total = 0 
        for batch in human_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            preds = linear_model(embeddings)
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        best_human_acc = correct/total
    outputs = {"train_acc": best_train_acc, "val_acc": best_val_acc,
                 "test_acc": best_test_acc, "human_acc": best_human_acc}
    return outputs 
        




