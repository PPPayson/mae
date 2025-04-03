import argparse
import numpy as np
import os
import torch
import random
import timm
import wandb
import src.util.misc as misc
from pathlib import Path
from src.models import models_autoreg
from src.models.models_encoder import LinearModel
from main_pretrain_autoreg import get_model as get_autoreg_model
from main_pretrain_dino import get_model as get_dino_model
from src.models import models_vit
from main_pretrain_autoreg import get_args_parser as model_get_args_parser
from src.engine_eval import eval_alignment, eval_co3d
from src.data.datasets import build_co3d_eval_loader, build_pretraining_dataset
from src.util.pos_embed import interpolate_pos_embed
torch.multiprocessing.set_sharing_strategy('file_system')

def get_model(args):
    if 'dino' in args.model:
        return get_dino_model(args)
    elif 'autoreg' in args.model:
        return get_autoreg_model(args)
    else:
        return models_vit.__dict__[args.model](
            num_classes=0,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

def get_args_parser():
    parser = model_get_args_parser()
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
        msg = model.load_state_dict(checkpoint['model'])
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

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        print(model)
        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    co3d_train_dataloader, co3d_val_dataloader, _, _ = build_co3d_eval_loader(args, None, True, (not args.dataset=='imgnet'))
    co3d_test_dataloader, imgnet_test_dataloader = build_co3d_eval_loader(args, None, False, (not args.dataset=='imgnet'))    
    
    if args.log_dir is not None and not args.no_wandb:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.login(key="50923956f844f2494110e5b6c020d31fa1072149")
        log_writer = misc.WandBLogger(log_dir=args.log_dir, args=args)
    else:
        log_writer = None
    outputs = eval_co3d(model, co3d_train_dataloader,
                co3d_val_dataloader, co3d_test_dataloader, 
                imgnet_test_dataloader, device, 0, num_epochs=args.eval_co3d_epochs, 
                batch_size=args.batch_size, learning_rate=5e-4, log_writer=log_writer,
                num_workers=args.num_workers, args=args, eval_align=True, timm_model=args.timm_model,
                linear_probe=((args.timm_model and args.linear_probe) or (not args.timm_model)), save_img=False)
    return outputs

if __name__ == "__main__":
    opts = get_args_parser().parse_args()
    os.makedirs(opts.output_dir, exist_ok=True)
    main(opts)