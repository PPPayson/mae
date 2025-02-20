# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import wandb
import torch
import timm
import torch.backends.cudnn as cudnn
from src.util.misc import WandBLogger
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm.optim.optim_factory as optim_factory
import src.util.misc as misc
from src.util.misc import NativeScalerWithGradNormCount as NativeScaler
from src.models import models_mae
from src.engine_pretrain import train_one_epoch
from src.engine_eval import eval_alignment, eval_co3d, eval_one_epoch
from src.data.datasets import build_co3d_eval_loader, build_pretraining_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_root', required=True, type=str, help='dataset root directory')

    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_val_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--clickmaps_path', default='./assets/co3d_val_processed.npz', type=str)
    parser.add_argument('--alignments_json', default='./assets/alignments.json', type=str)
    parser.add_argument('--clickmaps_human_path', default='./assets/human_ceiling_split_half_co3d_val.npz', type=str)
    parser.add_argument('--imgnet_clickmaps_path', default='./assets/jay_imagenet_for_co3d_val_0.1_processed.npz', type=str)
    parser.add_argument('--imgnet_clickmaps_human_path', default='./assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz', type=str)
    parser.add_argument('--imgnet2co3d_label', default='./assets/synset_to_co3d.npy', type=str)

    parser.add_argument('--not_pretrained', default=False, action='store_true')
    parser.add_argument('--video_dataset', default=False, action='store_true')
    parser.add_argument('--drop_path', default=0, type=float)
    parser.add_argument('--num_frames', default=4, type=int)
    parser.add_argument('--sampling_rate', default=4, type=int)
    parser.add_argument('--no_wandb', default=False, action='store_true')
    parser.add_argument('--binocular', default=False, action='store_true')
    parser.add_argument('--reverse_sequence', default=False, action='store_true')
    parser.add_argument('--data_length_divisor', default=1, type=int)
    parser.add_argument('--camera_param_dim', type=int, default=7)
    parser.add_argument('--camera_params', action='store_true', default=False, help='Train with Camera Parameters in the Decoder')
    parser.add_argument('--eval_co3d', action='store_true', default=False, help='Run evaluation on co3d classification and clickme maps')
    parser.add_argument('--eval_co3d_every', default=1, type=int)
    parser.add_argument('--eval_co3d_epochs', default=50, type=int)
    parser.add_argument('--eval_co3d_batch_size', default=512, type=int)
    parser.add_argument('--eval_co3d_lr', default=5e-4, type=float)
    parser.add_argument('--timm_pool', default=False, action='store_true')
    parser.add_argument("--img_data_path", default='', type=str)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                 pretrained=not args.not_pretrained, drop_path_rate=args.drop_path,
                                 epochs=args.epochs)

    model.to(device)
    args.mean = model.mean
    args.std = model.std
    args.mask_type = 'random'
    patch_size = model.patch_embed.patch_size

    args.window_size = (args.num_frames, args.input_size // patch_size[0], args.input_size // patch_size[1])
    print("Video Dataset", args.video_dataset)
    if args.video_dataset:
        dataset_train = build_pretraining_dataset(args)
        dataset_val = build_pretraining_dataset(args, is_train=False)
        co3d_train_dataloader, co3d_val_dataloader, _, _ = build_co3d_eval_loader(args, None, True)
        co3d_test_dataloader, imgnet_test_dataloader = build_co3d_eval_loader(args, None, False) 
    else:
        # simple augmentation
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std)])

                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.ImageFolder(os.path.join(args.img_data_path, 'train'), transform=transform_train)

        transform_val = transforms.Compose([
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
            ])
        dataset_val = datasets.ImageFolder(os.path.join(args.img_data_path, 'validation'), transform=transform_val)
        co3d_train_dataloader, co3d_val_dataloader, _, _ = build_co3d_eval_loader(args, transform_val, True)
        co3d_test_dataloader, imgnet_test_dataloader = build_co3d_eval_loader(args, transform_val, False)  
   

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        num_tasks=1
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)


    if global_rank == 0 and args.log_dir is not None and not args.no_wandb:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.login(key="50923956f844f2494110e5b6c020d31fa1072149")
        log_writer = WandBLogger(log_dir=args.log_dir, args=args)
        #log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    model_without_ddp = model
    # for p in model_without_ddp.encoder.parameters():
    #    p.requires_grad=False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        # model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    eval_co3d(model_without_ddp, co3d_train_dataloader,
                co3d_val_dataloader, co3d_test_dataloader, 
                imgnet_test_dataloader, device, 0, num_epochs=50, 
                batch_size=args.batch_size, learning_rate=5e-4, log_writer=log_writer,
                num_workers=args.num_workers, args=args, eval_align=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  
        # if epoch == 20:
        #     for p in model_without_ddp.encoder.parameters():
        #         p.requires_grad=True
        #     model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=False)             
        #     param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        #     optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, 
            start_steps=epoch*num_training_steps_per_epoch,
            args=args
        )
        eval_one_epoch(
            model, data_loader_val,
            device, epoch,
            log_writer=log_writer,
            start_steps=(epoch+1)*num_training_steps_per_epoch,
             args=args)
             
        eval_outputs = eval_co3d(model_without_ddp, co3d_train_dataloader,
                    co3d_val_dataloader, co3d_test_dataloader, 
                    imgnet_test_dataloader, device, epoch, num_epochs=50, 
                    batch_size=args.batch_size, learning_rate=5e-4, log_writer=log_writer,
                    num_workers=args.num_workers, args=args, eval_align=True)
        acc = eval_outputs['acc']

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
