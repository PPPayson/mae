# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import wandb
import torch
from torch import nn
import torch.distributed as dist
#from torch._six import inf
import math
import random
import io
import numpy as np
from datetime import timedelta
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from timm.utils import get_state_dict
from scipy.spatial.transform import Rotation as R
from timm.data.transforms import str_to_interp_mode
from timm.layers.helpers import to_2tuple
from src.util.dino import cancel_gradients_last_layer


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
class WandBLogger(object):
    def __init__(self, log_dir, args=None, project="multiviewMAE"):
        self.args = args
        import tempfile
        fixed_name = args.output_dir.split('/')[1]
        self.writer = wandb.init(project=project, 
                            entity="peisen_zhou",
                            dir=tempfile.gettempdir(),
                            config=args)
        wandb_random_string = self.writer.name
        new_run_name = f"{fixed_name}_{wandb_random_string}"
        self.writer.name = new_run_name
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
        # self.writer.log({}, commit=True)

    def update(self, head='scalar', step=None, commit=False, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if "frames" in k:
                # import pdb; pdb.set_trace()
                v = torch.cat(v)
                img_grid = make_grid(v, nrow=self.args.num_frames, normalize=not True, scale_each=not True)
                img_grid = F.to_pil_image(img_grid.clip(0,0.996))
                # img_grid = Image.fromarray((img_grid.movedim(0,2).cpu().numpy() * 255).astype(np.uint8))
                img_grid = wandb.Image(img_grid, caption=f"input _ pred _ label")
                self.writer.log({k: [img_grid]}, step = self.step if step is None else step, commit=commit)
                continue
            elif "maps" in k:
                k = head + '_' + k
                v = torch.cat(v, 0)
                v = torch.moveaxis(v, 3, 1)
                img_grid = make_grid(v, nrow=1, normalize=True, scale_each=False)
                img_grid = F.to_pil_image(img_grid.clip(0, 0.996))
                img_grid = wandb.Image(img_grid, caption=f"full _ topk _ topk_blur _ topk-half _ human _ img")
                self.writer.log({k: [img_grid]}, step = self.step if step is None else step, commit=commit)
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.log({head + "/" + k: v}, step = self.step if step is None else step, commit=commit)

    def flush(self):
        # self.writer.flush()
        pass

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t, dist.ReduceOp.AVG, async_op=False)
        #dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return
    print(args.gpu)
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, timeout=timedelta(minutes=30))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, freeze_last_layer=-1, epoch=0, model=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            cancel_gradients_last_layer(epoch, model, freeze_last_layer)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if math.isinf(norm_type):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


# def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
#     output_dir = Path(args.output_dir)
#     epoch_name = str(epoch)
#     if loss_scaler is not None:
#         checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
#         for checkpoint_path in checkpoint_paths:
#             to_save = {
#                 'model': model_without_ddp.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'scaler': loss_scaler.state_dict(),
#                 'args': args,
#             }

#             save_on_master(to_save, checkpoint_path)
#     else:
#         client_state = {'epoch': epoch}
#         model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def matrix_to_quaternion(matrix):
    r = R.from_matrix(matrix)
    return r.as_quat()

def matrix_to_euler(matrix):
    r = R.from_matrix(matrix)
    return r.as_euler("XYZ")

def euler_to_matrix(euler):
    r = R.from_euler("xyz", euler)
    return r.as_matrix()

def euler_to_quaternion(euler):
    r = R.from_euler("xyz", euler)
    return r.as_quat()

def quaternion_to_matrix(quat):
    r = R.from_quat(quat)
    return r.as_matrix()

def quaternion_to_euler(quat):
    r = R.from_quat(quat)
    return r.as_euler("xyz")

def camera_to_category(rot_mat, translation):
    euler = matrix_to_euler(rot_mat)
    euler_class =  np.any([np.all([euler>0, euler<=math.pi],axis=0), np.all([euler<=0, euler<=-math.pi], axis=0)], axis=0).astype(int)
    translation_class = (translation>0).astype(int)
    euler_class = 4*euler_class[0] + 2*euler_class[1] + euler_class[2]
    translation_class = 4*translation_class[0] + 2*translation_class[1] + translation_class[2]
    combined_class = euler_class + 8*translation_class
    return combined_class

def get_transform_center_crop(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)
    mean = data_config['mean']
    std = data_config['std']
    tf = []
    tf += [transforms.Resize(256)]
    tf += [transforms.CenterCrop(224)]
    tf += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tf)

def get_transform_wo_crop(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)
    mean = data_config['mean']
    std = data_config['std']
    interpolation = data_config['interpolation']
    tf = []
    tf += [transforms.Resize(input_size[0], interpolation=str_to_interp_mode(interpolation))]

    tf += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tf)


def get_cvm_attn_mask(q_len, num_frames, offset=0):
    q_len = q_len // num_frames
    attn_mask = torch.tril(torch.ones((num_frames, num_frames)), diagonal=offset).to(torch.bool)
    attn_mask = attn_mask.repeat_interleave(q_len, dim=1, output_size=q_len*num_frames).repeat_interleave(q_len, dim=0, output_size=q_len*num_frames)
    return attn_mask

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, linear_model_without_ddp=None, linear_optimizer=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if linear_model_without_ddp is not None:
        linear_weight = linear_model_without_ddp.state_dict()
        linear_optimizer_state = linear_optimizer.state_dict()
    else:
        linear_weight = None
        linear_optimizer_state = None
    if loss_scaler != None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'linear_model': linear_weight,
                'optimizer': optimizer.state_dict(),
                'linear_optimizer': linear_optimizer_state,
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema != None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema != None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, linear_model_without_ddp=None, linear_optimizer=None):
    output_dir = Path(args.output_dir)
    if loss_scaler != None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            if linear_model_without_ddp is not None and 'linear_model' in checkpoint.keys():
                linear_model_without_ddp.load_state_dict(checkpoint['linear_model'])
            if linear_optimizer is not None and 'lienar_optimizer' in checkpoint.keys():
                linear_optimizer.load_state_dict(checkpoint['linear_optimizer'])
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:  
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


# Origin: https://raw.githubusercontent.com/facebookresearch/jepa/3081b0ad7b9651373ccef40c1d46b62f46cb7146/src/models/utils/patch_embed.py
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        num_frames=8
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.tubelet_size = tubelet_size
        self.grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.img_size = to_2tuple(img_size)
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[1], patch_size[0]),
        )

    def forward(self, x, **kwargs):
        if len(x.shape) < 5:
            x = x[:, :, None, :, :]
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2) #NLC
        return x