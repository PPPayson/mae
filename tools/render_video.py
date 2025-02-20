import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import timm
import random
import math
import src.util.misc as misc
from src.data.datasets import build_pretraining_dataset, build_co3d_eval_loader
from main_pretrain_autoreg import get_args_parser as get_autoreg_args
from main_pretrain_autoreg import get_model as get_autoreg_model
from src.util.pos_embed import interpolate_pos_embed
from einops import rearrange
from typing import Iterable
from src.models.models_encoder import LinearModel, FullModel, LinearModelTimm
from src.data.co3d_dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from src.util.save_features import extract_features
from matplotlib import pyplot as plt
from torchvision.transforms import functional as tvF
from torchvision.utils import make_grid
import src.util.clickme_utils as clickme_utils
from PIL import Image


def get_args_parser():
    parser = get_autoreg_args()
    parser.add_argument('--timm_model', default=False, action='store_true')
    parser.add_argument('--linear_probe', default=False, action='store_true')
    parser.add_argument('--drop_rate', default=0., type=float)

    #DINO parameters
    parser.add_argument('--momentum_teacher', default=0.9995, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=15, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--pos_embed', type=str, default='1d_spatial', choices=['1d_spatial', '1d_temporal', '2d', '3d'])

    return parser


def get_model(args):
    if 'dino' in args.model:
        return get_dino_model(args)
    if 'autoreg' in args.model:
        return get_autoreg_model(args)


def train_head(model: torch.nn.Module, 
            train_data_loader: Iterable, 
            val_data_loader: Iterable, 
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
            timm_model=False,):
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
    linear_model.load_state_dict(best_model_state)
    full_model = FullModel(encoder, linear_model, pool=args.timm_pool, timm_model=timm_model)
    return full_model
def main(args):
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(args.model, args.linear_probe)
    if args.timm_model:
        if args.linear_probe:
            model = timm.create_model(args.model, pretrained=(not args.not_pretrained), num_classes=0, drop_rate=args.drop_rate)
        else:
            model = timm.create_model(args.model, pretrained=(not args.not_pretrained), num_classes=args.num_classes, drop_rate=args.drop_rate)
        config = timm.data.resolve_model_data_config(model)
        args.mean = config['mean']
        args.std = config['std']
        if args.ckpt_path != None:
            ckpt_dict = torch.load(args.ckpt_path)
            for k in list(ckpt_dict.keys()):
                if 'module.' in k:
                    n_k = k.replace('module.', '')
                    ckpt_dict[n_k] = ckpt_dict[k]
                    del ckpt_dict[k]
            model.load_state_dict(ckpt_dict)
        patch_size = model.patch_embed.patch_size
        args.window_size = (1, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size
        
    elif 'dino' in args.model:
        model = get_model(args)
        args.mean = model.mean
        args.std = model.std
        patch_size = model.teacher_encoder.patch_embed.patch_size
        args.window_size = (args.num_frames, args.input_size // patch_size, args.input_size // patch_size)
        args.patch_size = patch_size
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif 'autoreg' in args.model:
        if args.use_cce:
            args.decoder_num_classes = 768*16
        else:
            args.decoder_num_classes = 768
        model = get_model(args)
        args.mean = model.mean
        args.std = model.std
        patch_size = model.encoder.patch_embed.patch_size
        args.window_size = (args.num_frames, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        args.global_pool=False
        model = get_model(args)
        args.mean = model.mean
        args.std = model.std
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.ckpt_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    co3d_train_dataloader, co3d_val_dataloader, _, _ = build_co3d_eval_loader(args, None, True, (not args.dataset=='imgnet'))
    full_model = train_head(model,
                    co3d_train_dataloader,
                    co3d_val_dataloader,
                    args.device,
                    0,
                    num_epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    learning_rate=5e-4,  
                    num_workers=args.num_workers, 
                    args=args)

    args.num_frames = 49
    args.sampling_rate=1
    dataset_val = build_pretraining_dataset(args, is_train=False)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=misc.seed_worker,
    )

    # video_indices = list(range(1920, 2024)) + list(range(93, 251)) + list(range(400, 527)) + \
    #                  list(range(1036, 1067)) + (list(range(1259, 1483))) + list(range(1580, 1665)) + \
    #                     list(range(2748, 2895)) + 
    # video_indices = list(range(2024, 2084)) + list(range(2287, 2428))
    video_indices = [2755, 2309, 2033, 2001, 1922, 1581, 1418, 525, 433]
    for index in video_indices:
        video_index = index
        start_index = 0
        clip = dataset_val.clips[video_index]
        images, _ = dataset_val.load_frames(clip, start_index, video_index)
        process_data, _ = dataset_val.transform((images, None))
        process_data = process_data.view((dataset_val.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

        process_data = process_data.to(device)

        process_data = rearrange(process_data,'n t h w -> t n h w')
        process_data.requires_grad = True
        outputs = full_model(process_data)
        most_probable_classes = torch.argmax(outputs, dim=-1)
        most_probable_scores = outputs[torch.arange(outputs.size(0)), most_probable_classes]  # Shape: (batch_size,)
        saliency = torch.autograd.grad(outputs=most_probable_scores, inputs=process_data, 
                                            grad_outputs=torch.ones_like(most_probable_scores),
                                            create_graph=False)[0]
        saliency = torch.amax(saliency.abs(), dim=1).detach().cpu()
        kernel = clickme_utils.circle_kernel(21, 21).to(device)

        for i, img in enumerate(process_data):
            img = img.clone().detach().cpu().numpy().squeeze()
            img = np.moveaxis(img, 0, -1)
            hmp = saliency[i].clone()[None, :, :]
            hmp = clickme_utils.gaussian_blur(hmp.to(device).unsqueeze(0), kernel)
            # Double convolve
            full_saliency = clickme_utils.gaussian_blur(hmp, kernel).squeeze()
            hmp = hmp.detach().cpu().numpy().squeeze()
            hmp = (hmp - hmp.min()) / (hmp.max() - hmp.min())
            img = (img - img.min()) / (img.max() - img.min())
            f = plt.figure(figsize=(4.48, 2.24), dpi=300)
            plt.subplot(1, 2, 1)
            plt.imshow(hmp.squeeze())
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(img.squeeze())
            plt.axis('off')
            f.tight_layout(pad=0, h_pad=0, w_pad=0)
            f.canvas.draw()
            buf = f.canvas.buffer_rgba()
            ncols, nrows = f.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(nrows, ncols, 4)
            image = torch.unsqueeze(torch.Tensor(image), 0)
            # print(image.shape)
            # image = image[:, int(math.floor(image.shape[1]/6)):int(image.shape[1] - math.floor(image.shape[1]/6)), :, :]
            # print(image.shape)
            v = torch.moveaxis(image, 3, 1)
            img_grid = make_grid(v, nrow=1, normalize=True, scale_each=False)
            img_grid = tvF.to_pil_image(img_grid.clip(0, 0.996))
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            img_name = f'{str(i).zfill(3)}.png'
            # array_name = f'{str(i).zfill(3)}.npy'
            os.makedirs(os.path.join(output_dir, str(video_index).zfill(4), 'img'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, str(video_index).zfill(4), 'hmp'), exist_ok=True)
            img_grid.save(os.path.join(output_dir, str(video_index).zfill(4), 'hmp', f'hmp_{img_name}'))
            # np.save(os.path.join(output_dir, str(video_index).zfill(4),array_name), hmp)
            # img = Image.fromarray((img * 255).astype(np.uint8))
            # img.save(os.path.join(output_dir, str(video_index).zfill(4), 'img', f'img_{img_name}'))
            plt.close()
            # print(output_dir)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)