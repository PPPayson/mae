import math
import copy
import numpy as np
from typing import Iterable
import torch
import torch.nn as nn
import timm
import os
import json
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
from models_2D import LinearModel, FullModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.co3d_dataset import EmbeddingDataset
from data.datasets import build_co3d_eval_loader
from scipy.stats import spearmanr
from util.save_features import extract_features
from torchvision.transforms import functional as tvF
from matplotlib import pyplot as plt
import submodules.clickme_processing.src.utils as clickme_utils
import util.misc as misc
import util.lr_sched as lr_sched
import utils 

def eval_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, 
                    epoch: int, log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'EVAL epoch: [{}]'.format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter
    mean = torch.as_tensor(args.mean).to(device)[None, :, None, None]
    std = torch.as_tensor(args.std).to(device)[None, :, None, None]

    with torch.no_grad():
        for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss, pred, _ = model(samples, mask_ratio=args.mask_ratio)
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
    if log_writer is not None:
        log_writer.set_step()        
        reconstruction = pred
        log_writer.update(loss=metric_logger.loss.avg, head="val")
        reconstruction = rearrange(reconstruction, 'b n (p c) -> b n p c', c=3)
        reconstruction = rearrange(reconstruction, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=16, p2=16, w=14, h=14)
        reconstruction = reconstruction * std + mean        
        log_writer.update(valframes=[reconstruction[0]], head="val")
        log_writer.update(commit=True, grad_norm=0, head="val")

def eval_co3d(model: torch.nn.Module, train_data_loader: Iterable, val_data_loader: Iterable, test_data_loader: Iterable, imgnet_loader: Iterable, device: torch.device, epoch: int, num_epochs: int, 
                batch_size: int, learning_rate=5e-4, log_writer=None, num_workers=16, args=None, eval_align=True):
    train_features, train_labels = extract_features(model, train_data_loader, device, False)
    val_features, val_labels = extract_features(model, val_data_loader, device, False)

    metric_logger = misc.MetricLogger(delimiter="   ")
    header = f'Co3D EVAL'
    print_freq = 10
    
    train_dataset = EmbeddingDataset(train_features, train_labels)
    val_dataset = EmbeddingDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    linear_model = LinearModel(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy())), dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_acc = 0
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
        acc = 100 * correct/total
        best_acc = max(acc, best_acc)
 
        metric_logger.update(train_loss=avg_train_loss)
        metric_logger.update(val_loss=avg_val_loss)
        metric_logger.update(acc=acc)
        metric_logger.synchronize_between_processes()

    if log_writer is not None:
            log_writer.set_step()        
            log_writer.update(val_acc=best_acc, head='co3d_eval')
            log_writer.update(epoch=epoch, head='co3d_eval')

    #TODO Use model with best val acc for alignment and test
    if eval_align:
        full_model = FullModel(model, linear_model, pool=False)
        avg_test_acc, alignment_scores, null_scores = eval_alignment(full_model, test_data_loader, device, log_writer, list(range(10)), args)
        imgnet_acc, imgnet_align, imgnet_null = eval_alignment(full_model, imgnet_loader, device, log_writer, list(range(10)), args)
        
        if log_writer is not None:
            log_writer.update(test_acc=avg_test_acc, head='co3d_eval')
            
            log_writer.update(full_auc=np.mean(alignment_scores['full_auc']), head='co3d_eval')
            log_writer.update(topk_auc=np.mean(alignment_scores['topk_auc']), head='co3d_eval')
            log_writer.update(halfk_auc=np.mean(alignment_scores['halfk_auc']), head='co3d_eval')
            log_writer.update(full_null_auc=np.mean(null_scores['full_auc']), head='co3d_eval_null')
            log_writer.update(topk_null_auc=np.mean(null_scores['topk_auc']), head='co3d_eval_null')
            log_writer.update(halfk_null_auc=np.mean(null_scores['halfk_auc']), head='co3d_eval_null')
            log_writer.update(full_spearman=np.mean(alignment_scores['full_spearman']), head='co3d_eval')
            log_writer.update(topk_spearman=np.mean(alignment_scores['topk_spearman']), head='co3d_eval')
            log_writer.update(halfk_spearman=np.mean(alignment_scores['halfk_spearman']), head='co3d_eval')
            log_writer.update(full_null_spearman=np.mean(null_scores['full_spearman']), head='co3d_eval_null')
            log_writer.update(topk_null_spearman=np.mean(null_scores['topk_spearman']), head='co3d_eval_null')
            log_writer.update(halfk_null_spearman=np.mean(null_scores['halfk_spearman']), head='co3d_eval_null')

            log_writer.update(test_acc=imgnet_acc, head='imgnet_eval')
            log_writer.update(full_auc=np.mean(imgnet_align['full_auc']), head='imgnet_eval')
            log_writer.update(topk_auc=np.mean(imgnet_align['topk_auc']), head='imgnet_eval')
            log_writer.update(halfk_auc=np.mean(imgnet_align['halfk_auc']), head='imgnet_eval')
            log_writer.update(full_null_auc=np.mean(imgnet_null['full_auc']), head='imgnet_eval_null')
            log_writer.update(topk_null_auc=np.mean(imgnet_null['topk_auc']), head='imgnet_eval_null')
            log_writer.update(halfk_null_auc=np.mean(imgnet_null['halfk_auc']), head='imgnet_eval_null')
            log_writer.update(full_spearman=np.mean(imgnet_align['full_spearman']), head='imgnet_eval')
            log_writer.update(topk_spearman=np.mean(imgnet_align['topk_spearman']), head='imgnet_eval')
            log_writer.update(halfk_spearman=np.mean(imgnet_align['halfk_spearman']), head='imgnet_eval')
            log_writer.update(full_null_spearman=np.mean(imgnet_null['full_spearman']), head='imgnet_eval_null')
            log_writer.update(topk_null_spearman=np.mean(imgnet_null['topk_spearman']), head='imgnet_eval_null')
            log_writer.update(halfk_null_spearman=np.mean(imgnet_null['halfk_spearman']), head='imgnet_eval_null')
    return

def eval_alignment(full_model:torch.nn.Module, test_data_loader: Iterable, 
                device: torch.device, log_writer, visualize, args, return_acc=True, 
                kernel_size=21, kernel_sigma=21):

    #TODO Compute halfk score with half human maps
    # Check if model maps have an offset
    # Think of better way to do topk filter
    full_model.eval()
    sample_maps = []
    with open(args.alignments_json, 'r') as f:
        alignments = json.load(f)
    # with open(args.stats_json, 'r') as f:
    #     num_pos = json.load(f)
    #     half_num_pos = num_pos['half_num_pos']
    #     num_pos = num_pos['num_pos']
    criterion = nn.CrossEntropyLoss()
    #TODO Use model with best val acc for alignment and test
    auc_ceiling = alignments[test_data_loader.dataset.dataset_name]
    spearman_ceiling = test_data_loader.dataset.ceiling
    alignment_scores = {'full_auc':[], 'topk_auc':[], 'halfk_auc': [], 'full_spearman':[], 'topk_spearman':[], 'halfk_spearman': []}
    null_scores = {'full_auc':[], 'topk_auc':[], 'halfk_auc': [], 'full_spearman':[], 'topk_spearman':[], 'halfk_spearman': []}
    all_test_acc = []


    #kernel = utils.gaussian_kernel(size=21, sigma=math.sqrt(21)).to(device)
    kernel = clickme_utils.circle_kernel(kernel_size, kernel_sigma).to(device)

    # Evaluate each image
    for i, batch in tqdm(enumerate(test_data_loader)):
        imgs, hmps, labels, img_names, cat = batch
        imgs, hmps, labels = imgs.to(device), hmps.to(device), labels.to(device)
        img_name = img_names[0]
        cat = cat[0]

        # Select a random hmp for null score
        sub_vec = np.where(np.array(test_data_loader.dataset.categories) != cat)[0]
        random_idx = np.random.choice(sub_vec)
        random_hmps = test_data_loader.dataset[random_idx]
        random_hmps = torch.unsqueeze(torch.Tensor(random_hmps[1]), 0)

        # Save a copy of the img for visualization
        if len(visualize)>0 and i in visualize:
            img = imgs.clone().detach().cpu().numpy().squeeze()
            img = np.moveaxis(img, 0, -1)
            img = img*args.std + args.mean
            img = np.uint8(255*img)

        # Get accuracy and loss
        if return_acc:
            outputs = full_model(imgs)
            # loss = criterion(outputs, labels)

            test_acc = utils.accuracy(outputs, labels)[0].item()
            all_test_acc.append(test_acc)

        # Get saliency map with smoothgrad
        saliency = torch.Tensor(utils.batch_smooth_grad(full_model, imgs))
        if saliency.shape[-1] != 224:
            saliency = F.interpolate(saliency.unsqueeze(0), size=(224, 224), mode="bilinear").to(torch.float32)

        # Get average hmps and average half hmps
        if test_data_loader.dataset.dataset_name == "imgnet":
            hmps = tvF.resize(hmps, 256)

        hmps = tvF.center_crop(hmps, (224, 224))
        hmps_indices = list(range(hmps.shape[1]))
        random_index = np.random.choice(hmps_indices, int(len(hmps_indices)/2), replace=False)
        half_hmps = hmps[:, random_index, :, :]
        half_hmps = half_hmps.mean(1)
        hmps = hmps.mean(1)
        if test_data_loader.dataset.dataset_name == "imgnet":
            random_hmps = tvF.resize(random_hmps, 256)

        random_hmps = tvF.center_crop(random_hmps, (224, 224))
        hmps_indices = list(range(random_hmps.shape[1]))
        random_half_hmps = random_hmps[:, np.random.choice(hmps_indices, int(len(hmps_indices)/2), replace=False), :, :].mean(1)
        random_hmps = random_hmps.mean(1)

        hmps = (hmps - hmps.min()) / (hmps.max() - hmps.min())
        half_hmps = (half_hmps - half_hmps.min())/ (half_hmps.max() - half_hmps.min())
        random_hmps = (random_hmps - random_hmps.min()) / (random_hmps.max() - random_hmps.min())
        random_half_hmps = (random_half_hmps - random_half_hmps.min()) / (random_half_hmps.max() - random_half_hmps.min())

        # Get topk model saliency points and half top k to match sparisty of hmps and half hmps
        full_saliency = saliency
        topk_saliency = saliency.clone()
        halfk_saliency = saliency.clone()

        topk_saliency = utils.gaussian_blur(topk_saliency.to(device).unsqueeze(0), kernel)
        halfk_saliency = utils.gaussian_blur(halfk_saliency.to(device).unsqueeze(0), kernel)
        full_saliency = utils.gaussian_blur(full_saliency.to(device).unsqueeze(0), kernel)
        # Double convolve
        topk_saliency = utils.gaussian_blur(topk_saliency, kernel).squeeze()
        halfk_saliency = utils.gaussian_blur(halfk_saliency, kernel).squeeze()
        full_saliency = utils.gaussian_blur(full_saliency, kernel).squeeze()

        k = torch.sum(hmps>0)
        flat_saliency = topk_saliency.flatten()
        topk, indices = torch.topk(flat_saliency, k)
        thresh_value = topk[-1]
        topk_saliency[topk_saliency<thresh_value] = 0

        k = torch.sum(half_hmps>0)
        flat_saliency = halfk_saliency.flatten()
        topk, indices = torch.topk(flat_saliency, k)
        thresh_value = topk[-1]
        halfk_saliency[halfk_saliency<thresh_value] = 0

        full_saliency = full_saliency.detach().cpu().numpy()
        topk_saliency = topk_saliency.detach().cpu().numpy()
        halfk_saliency = halfk_saliency.detach().cpu().numpy()
        hmps = hmps.detach().cpu().numpy()
        random_hmps = random_hmps.detach().cpu().numpy()
        half_hmps = half_hmps.detach().cpu().numpy()
        random_half_hmps = random_half_hmps.detach().cpu().numpy()

        # Normalize
        full_saliency = (full_saliency - full_saliency.min())/(full_saliency.max() - full_saliency.min())
        topk_saliency = (topk_saliency - topk_saliency.min())/(topk_saliency.max() - topk_saliency.min())
        halfk_saliency = (halfk_saliency - halfk_saliency.min())/(halfk_saliency.max() - halfk_saliency.min())

        # Compute AUC
        full_auc = clickme_utils.compute_AUC(full_saliency, hmps)
        topk_auc = clickme_utils.compute_AUC(topk_saliency, hmps)
        halfk_auc = clickme_utils.compute_AUC(halfk_saliency, half_hmps)
        
        # Compute null AUC vs random map
        full_null_auc = clickme_utils.compute_AUC(full_saliency, random_hmps)
        topk_null_auc = clickme_utils.compute_AUC(topk_saliency, random_hmps)
        halfk_null_auc = clickme_utils.compute_AUC(halfk_saliency, random_half_hmps)

        # Compute spearman
        full_spearman = clickme_utils.compute_spearman_correlation(full_saliency, hmps)
        topk_spearman = clickme_utils.compute_spearman_correlation(topk_saliency, hmps)
        halfk_spearman = clickme_utils.compute_spearman_correlation(halfk_saliency, half_hmps)

        full_null_spearman = clickme_utils.compute_spearman_correlation(full_saliency, random_hmps)
        topk_null_spearman = clickme_utils.compute_spearman_correlation(topk_saliency, random_hmps)
        halfk_null_spearman = clickme_utils.compute_spearman_correlation(halfk_saliency, random_half_hmps)

        #Scale to human ceiling percentage
        topk_auc /= auc_ceiling
        full_auc /= auc_ceiling
        halfk_auc /= auc_ceiling
        
        topk_null_auc /= auc_ceiling
        full_null_auc /= auc_ceiling
        halfk_null_auc /= auc_ceiling

        topk_spearman /= spearman_ceiling
        full_spearman /= spearman_ceiling
        halfk_spearman /= spearman_ceiling
        
        topk_null_spearman /= spearman_ceiling
        full_null_spearman /= spearman_ceiling
        halfk_null_spearman /= spearman_ceiling



        alignment_scores['full_auc'].append(full_auc)
        alignment_scores['topk_auc'].append(topk_auc)
        alignment_scores['halfk_auc'].append(halfk_auc)

        null_scores['full_auc'].append(full_null_auc)
        null_scores['topk_auc'].append(topk_null_auc)
        null_scores['halfk_auc'].append(halfk_null_auc)

        alignment_scores['full_spearman'].append(full_spearman)
        alignment_scores['topk_spearman'].append(topk_spearman)
        alignment_scores['halfk_spearman'].append(halfk_spearman)

        null_scores['full_spearman'].append(full_null_spearman)
        null_scores['topk_spearman'].append(topk_null_spearman)
        null_scores['halfk_spearman'].append(halfk_null_spearman)

        # Save image for wandb log
        if len(visualize)>0 and i in visualize:
            hmps_img = hmps.squeeze()
            f = plt.figure()
            plt.subplot(1, 6, 1)
            plt.imshow(full_saliency.squeeze())
            plt.axis("off")
            plt.subplot(1, 6, 2)
            plt.imshow(topk_saliency.squeeze())
            plt.axis("off")
            plt.subplot(1, 6, 3)
            plt.imshow(halfk_saliency.squeeze())
            plt.axis("off")
            plt.subplot(1, 6, 4)
            hmps_img = (hmps_img - np.min(hmps_img))/np.max(hmps_img)
            plt.imshow(hmps_img)
            plt.axis("off")
            plt.subplot(1, 6, 5)
            plt.imshow(img)
            plt.axis("off")

            f.tight_layout(pad=0)
            f.canvas.draw()
            buf = f.canvas.buffer_rgba()
            ncols, nrows = f.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
            image = torch.unsqueeze(torch.Tensor(image), 0)
            image = image[:, int(math.floor(image.shape[1]/4)):int(image.shape[1] - math.floor(image.shape[1]/4)), :, :]
            sample_maps.append(image)
            plt.close()

    if return_acc:      
        avg_test_acc = sum(all_test_acc)/float(len(all_test_acc))
    else:
        avg_test_acc = 0
    if log_writer is not None:
        log_writer.update(heatmaps=sample_maps, head=f"{test_data_loader.dataset.dataset_name}_eval")
    return avg_test_acc, alignment_scores, null_scores


