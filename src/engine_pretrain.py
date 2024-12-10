# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import src.util.misc as misc
import src.util.lr_sched as lr_sched
from einops import rearrange

def autoreg_train_one_epoch(model: torch.nn.Module, 
                            data_loader: Iterable,
                            optimizer: torch.optim.Optimizer,
                            device: torch.device,
                            epoch: int, loss_scaler,
                            max_norm: float = 0,
                            patch_size: int = 16,
                            normalize_target: bool = False,
                            log_writer=None, 
                            start_steps=0,
                            use_cce=True, 
                            n_frames=4, 
                            camera_params_enabled=False,
                            schedule_free=False, 
                            categorical_camera=False, 
                            alpha=(0.5, 0.5), 
                            linear_model=None, 
                            linear_optimizer=None, 
                            feature_loss=False, 
                            args=None):
    model.train()
    if linear_model is not None:
        linear_model.train()
    if schedule_free:
        optimizer.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if use_cce:
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.MSELoss()

    optimizer.zero_grad()
    mean = torch.as_tensor(args.mean).to(device)[None, :, None, None, None]
    std = torch.as_tensor(args.std).to(device)[None, :, None, None, None]
    params = model.parameters()
    if linear_model is not None:
        params = list(params) + list(linear_model.parameters())
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + data_iter_step
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
        with torch.no_grad():
            unnorm_videos = videos * std + mean
            videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=1, p1=patch_size, p2=patch_size)
            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(videos, true_camera_params)
            pred_frames = outputs['pred_frames']
            if use_cce:
                # labels_long = ((labels - torch.min(labels, dim=-2, keepdim=True).values) / (torch.max(labels, dim=-2, keepdim=True).values - torch.min(labels, dim=-2, keepdim=True).values + 1e-7)).to(torch.long)
                labels_long = (labels*15).to(torch.long)
                pred_frames = rearrange(pred_frames, 'b (t n) (p c) -> b t c n p', c=16, t=n_frames-1)
                pred_frames = pred_frames[:, 1:, :, :, :]
                pred_frames = rearrange(pred_frames, 'b t c n p -> b c (t n) p', c=16, t=n_frames-2)
                loss = loss_func(input=pred_frames, target=labels_long)

            else:
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
                features = rearrange(features[:, :n_frames-1, :], 'b t c -> (b t) c')
                features = F.softmax(features, dim=1)
                f_loss = F.cross_entropy(new_features, features)/math.log(1000)
                f_loss_value = f_loss.item()
                loss += alpha[1]*f_loss
                metric_logger.update(feature_loss=f_loss_value)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=params, create_graph=is_second_order)
        
        metric_logger.synchronize_between_processes()
        if log_writer is not None:
            log_writer.set_step(it)
            if it%100==0:
                reconstruction=videos_patch.clone()
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
                log_writer.update(frames=[unnorm_videos[0].transpose(0, 1), reconstruction[0], masked_inp[0]], head="frames")

            if it%40==0:
                log_writer.update(loss=metric_logger.loss.value, head="train")
                log_writer.update(recon_loss=metric_logger.recon_loss.value, head="train")
                log_writer.update(epoch=epoch, head="train")
                if camera_params_enabled:
                    log_writer.update(cam_loss=metric_logger.cam_loss.value, head='train')
                if feature_loss and linear_model is not None:
                    log_writer.update(feature_loss=metric_logger.feature_loss.value, head='train')
                log_writer.update(commit=True, grad_norm=grad_norm, head="opt")

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
    
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
                    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    mean = torch.as_tensor(args.mean).to(device)[None, :, None, None]
    std = torch.as_tensor(args.std).to(device)[None, :, None, None]

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            log_writer.set_step()
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value_reduce, head="train")
            log_writer.update(lr=lr, head="train")
            #log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            #log_writer.add_scalar('lr', lr, epoch_1000x)
    if log_writer is not None:
        reconstruction = pred
        reconstruction = rearrange(reconstruction, 'b n (p c) -> b n p c', c=3)
        reconstruction = rearrange(reconstruction, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=16, p2=16, w=14, h=14)
        reconstruction = reconstruction * std + mean
        log_writer.update(frames=[reconstruction[0]], head="train", commit=True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}