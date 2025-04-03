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
import numpy as np
import os
import time
from pathlib import Path
import wandb
import torch
import timm
import random
from src.util.misc import WandBLogger
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.util.optim_factory import create_optimizer
import src.util.misc as misc
from src.util.misc import NativeScalerWithGradNormCount as NativeScaler
from src.models.methods import models_autoreg
from src.models.backbones.models_encoder import LinearModel
from src.engine_pretrain import autoreg_train_one_epoch
from src.engine_eval import eval_alignment, eval_co3d, autoreg_eval_one_epoch
from src.data.datasets import build_co3d_eval_loader, build_pretraining_dataset
from src.util.save_features import load_logits

def get_args_parser():
    parser = argparse.ArgumentParser('Autoregressive Vision Model pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--not_pretrained', default=False, action='store_true')
    parser.add_argument('--eval_co3d', action='store_true', default=False, help='Run evaluation on co3d classification and clickme maps')
    parser.add_argument('--eval_co3d_every', default=1, type=int)
    parser.add_argument('--eval_co3d_epochs', default=50, type=int)
    parser.add_argument('--eval_co3d_batch_size', default=512, type=int)
    parser.add_argument('--eval_co3d_lr', default=5e-4, type=float)

    # Model parameters
    parser.add_argument('--ckpt_path', type=str, help='Model checkpoint file path')
    parser.add_argument('--model', default='pretrain_videomae_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--timm_pool', default=False, action='store_true')
    parser.add_argument('--decoder_pos_embed', type=str, default='1d_spatial', choices=['1d_spatial', '1d_temporal', '2d', '3d', 'learned_3d'])
    parser.add_argument('--decoder_cls', default=False, action='store_true', help='Use cls token in the decoder')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'causal', 'causal_interpol', 'autoregressive'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--mask_ratio_var', default=0, type=float, nargs='+')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--single_video', default=False, action='store_true')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.)
    parser.add_argument('--normalize_target', default=True, type=bool, help='normalized the target patch pixels')
    parser.add_argument('--no-normalize_target', dest='normalize_target', action='store_false')
    parser.add_argument('--decoder_camera_dropout', type=float, default=0.0, help='dropout rate for the camera pose decoder')
    parser.add_argument('--tubelet_size', default=1, type=int)

    # Reverse the traversal order
    parser.add_argument('--reverse_sequence', default=True, type=bool, help='Reverse the sequence of frame traversal')
    parser.add_argument('--no-reverse_sequence', dest='reverse_sequence', action='store_false')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument("--schedule_free", action="store_true", default=False, help="Whehter to use schedule free optimizer")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--shared_transform', default=False, action='store_true')
    parser.add_argument('--photo_trasnform', default=False, action='store_true')

    # Dataset parameters
    parser.add_argument('--dataset', default='co3d', type=str, choices=['co3d', 'mvimgnet', 'co3d_mvimgnet', 'imgnet', 'co3d_video'])
    parser.add_argument('--data_root', required=True, type=str, help='dataset root directory')
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--data_val_path', default='/path/to/list_kinetics-400', type=str,
                        help='eval dataset path')
    parser.add_argument('--features_path', default='./features', type=str, help='path to saved encoder features')
    parser.add_argument('--stats_json', default='./assets/click_stats.json', type=str)
    parser.add_argument('--clickmaps_path', default='./assets/co3d_val_processed.npz', type=str)
    parser.add_argument('--alignments_json', default='./assets/alignments.json', type=str)
    parser.add_argument('--clickmaps_human_path', default='./assets/human_ceiling_split_half_co3d_val.npz', type=str)
    parser.add_argument('--imgnet_clickmaps_path', default='./assets/jay_imagenet_for_co3d_val_0.1_processed.npz', type=str)
    parser.add_argument('--imgnet_clickmaps_human_path', default='./assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz', type=str)
    parser.add_argument('--imgnet2co3d_label', default='./assets/synset_to_co3d.npy', type=str)

    parser.add_argument('--feature_loss', action='store_true', default=False, help='Use original encoder features as additional supervision during fine-tuning')
    parser.add_argument('--alpha_f', default=0.5, type=float)
    parser.add_argument('--alpha_c', default=0.5, type=float)
    parser.add_argument('--num_classes', default=51, type=int)
    parser.add_argument('--binocular', action='store_true', default=False)
    parser.add_argument('--no-binocular', dest='binocular', action='store_false')
    parser.add_argument('--data_length_divisor', default=1, type=int)
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 4)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=not True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #CCE loss
    parser.add_argument('--use_cce', default=False, action='store_true',
                        help='Use CCE Loss')
    parser.add_argument('--categorical_camera', default=False, action='store_true', help='Use 64 categories for camera poses')

    # Camera Parameters
    parser.add_argument('--camera_param_dim', type=int, default=7)
    parser.add_argument('--camera_params', action='store_true', default=False, help='Train with Camera Parameters in the Decoder')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    return parser

def get_model(args):
    model = models_autoreg.__dict__[args.model](
        pretrained=not args.not_pretrained,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_num_classes=args.decoder_num_classes,
        use_checkpoint=args.use_checkpoint,
        camera_params_enabled=args.camera_params,
        ckpt_path=args.ckpt_path,
        attn_drop_rate = args.attn_drop_rate,
        camera_param_dim=args.camera_param_dim,
        categorical_camera=args.categorical_camera,
        num_frames = args.num_frames,
        decoder_pos_embed = args.decoder_pos_embed,
        decoder_cls = args.decoder_cls,
        timm_pool = args.timm_pool           
    )

    return model


def main(args):
    misc.init_distributed_mode(args)
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.benchmark = True
    if args.use_cce:
        args.decoder_num_classes = 768*16
    else:
        args.decoder_num_classes = 768

    model = get_model(args)
    args.mean = model.mean
    args.std = model.std

    if args.feature_loss:
        linear_model = LinearModel(model.encoder.embed_dim, args.num_classes)
        linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
    else:
        linear_model = None
        linear_optimizer = None
    patch_size = model.encoder.patch_embed.patch_size
    args.window_size = (args.num_frames, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    
    dataset_train = build_pretraining_dataset(args)
    dataset_val = build_pretraining_dataset(args, is_train=False)

    data_config = timm.data.resolve_model_data_config(model.encoder.model)
    # transform = misc.get_transform_center_crop(data_config)

    co3d_train_dataloader, co3d_val_dataloader, _, _ = build_co3d_eval_loader(args, None, True)
    co3d_test_dataloader, imgnet_test_dataloader = build_co3d_eval_loader(args, None, False)    

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=seed
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=seed
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)


    if global_rank == 0 and args.log_dir is not None and not args.no_wandb:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.login(key="50923956f844f2494110e5b6c020d31fa1072149")
        log_writer = WandBLogger(log_dir=args.log_dir, args=args)
    else:
        log_writer = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=misc.seed_worker,
        persistent_workers=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=misc.seed_worker,
    )

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
    
    if args.feature_loss:
        linear_model.to(device)
        linear_model_without_ddp = linear_model
    else:
        linear_model_without_ddp = None
    # define the model
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {} M'.format(n_parameters / 1e6))
    # for idx, (name, param) in enumerate(model_without_ddp.named_parameters()):
    #     print(f"Index: {idx}, Name: {name}, Shape: {param.shape}")

    eff_batch_size = args.batch_size * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("effective batch size: %d" % eff_batch_size)
    model_without_ddp = model
    # for p in model_without_ddp.encoder.parameters():
    #     p.requires_grad=False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        if args.feature_loss:
            linear_model = torch.nn.parallel.DistributedDataParallel(linear_model, device_ids=[args.gpu], find_unused_parameters=False)
            linear_model_without_ddp = linear_model.module


    if linear_model is not None:
        model_opt = (model, linear_model)
    else:
        model_opt = model
    if args.schedule_free:
        args.opt = 'adamWScheduleFree'
        if args.warmup_epochs > 0 and args.warmup_steps <= 0:
            args.warmup_steps = args.warmup_epochs * num_training_steps_per_epoch
        optimizer = create_optimizer(args, model_opt, filter_bias_and_bn=False)
    else:
        optimizer = create_optimizer(args, model_opt)    
    loss_scaler = NativeScaler()
    misc.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp, 
        optimizer=optimizer, 
        loss_scaler=loss_scaler, 
        linear_model_without_ddp=linear_model_without_ddp, 
        linear_optimizer=linear_optimizer)
    if args.feature_loss:
        t_features, t_labels, t_names, v_features, v_labels, v_names = load_logits(model_without_ddp.encoder, args)
        dataset_train.set_logits(t_features.cpu(), torch.squeeze(t_labels.cpu()))
        dataset_val.set_logits(v_features.cpu(), torch.squeeze(v_labels.cpu()))
    
    torch.cuda.empty_cache()


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    # acc = eval_co3d(model_without_ddp, co3d_train_dataloader,
    #             co3d_val_dataloader, co3d_test_dataloader, 
    #             imgnet_test_dataloader, device, -1, num_epochs=args.eval_co3d_epochs, 
    #             batch_size=args.batch_size, learning_rate=5e-4, log_writer=log_writer,
    #             num_workers=args.num_workers, args=args, eval_align=True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)             

        train_stats = autoreg_train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            model_without_ddp,
            co3d_train_dataloader,
            co3d_val_dataloader,
            co3d_test_dataloader,
            imgnet_test_dataloader,
            max_norm=args.clip_grad,
            start_steps=epoch*num_training_steps_per_epoch,
            patch_size=patch_size[0],
            use_cce=args.use_cce,
            n_frames=args.num_frames,
            schedule_free=args.schedule_free,
            linear_model=linear_model,
            feature_loss=args.feature_loss,
            log_writer=log_writer,
            camera_params_enabled=args.camera_params,
            categorical_camera=args.categorical_camera,
            alpha=(args.alpha_c, args.alpha_f),
            args=args
        )

        val_stats = autoreg_eval_one_epoch(
            model, data_loader_val,
            device, epoch,
            log_writer=log_writer,
            patch_size=patch_size[0],
            start_steps=(epoch+1)*num_training_steps_per_epoch,
            use_cce=args.use_cce,
            n_frames=args.num_frames,
            camera_params_enabled=args.camera_params,
            categorical_camera=args.categorical_camera,
            linear_model=linear_model,
            feature_loss=args.feature_loss,
            alpha=(args.alpha_c, args.alpha_f),
            args=args)
        eval_outputs = eval_co3d(model_without_ddp, co3d_train_dataloader,
                    co3d_val_dataloader, co3d_test_dataloader, 
                    imgnet_test_dataloader, device, epoch, num_epochs=args.eval_co3d_epochs, 
                    batch_size=args.batch_size, learning_rate=5e-4, log_writer=log_writer,
                    num_workers=args.num_workers, args=args, eval_align=True)
        acc = eval_outputs['acc']
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            if misc.is_main_process():
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, linear_model_without_ddp=linear_model_without_ddp, linear_optimizer=linear_optimizer)
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, linear_model_without_ddp=linear_model_without_ddp, linear_optimizer=linear_optimizer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str),)
    if misc.is_main_process():
        print(f'Best Acc {best_acc}')
        print(f'Best Epoch {best_epoch}')

if __name__ == '__main__':
    opts = get_args_parser().parse_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
