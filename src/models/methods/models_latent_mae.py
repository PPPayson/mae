from functools import partial
from itertools import chain
import timm
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from src.models.backbones.models_encoder import EncoderIBOT
from src.models.methods.models_ibot import iBOTHead

class LatentMAE(nn.Module):
    def __init__(self,
                encoder_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
                encoder_embed_dim=1024,
                out_dim=65536,
                patch_out_dim=-1,
                warmup_teacher_temp=0.04,
                warmup_teacher_patch_temp=0.04,
                teacher_temp=0.04,
                teacher_patch_temp=0.04,
                warmup_teacher_temp_epochs=10,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                pretrained=False,
                camera_params_enabled=False,
                camera_param_dim = 7,
                camera_categories = 64,
                categorical_camera = True,
                num_frames = 4,
                decoder_cls = False,
                timm_pool = False,
                epochs = 30,
                pos_embed='2d',
                teacher_3d = False,
                student_3d = False,
                global_ncrops = 2,
                local_ncrops = 8,
                lambda1=1.0,
                lambda2=1.0,
                pred_start_epoch=0,
                use_masked_im_modeling=True,
                **kwargs
                ):
        super().__init__()
        print("Pretrained", pretrained)
        print('Pos type', pos_embed)
        print("Encoder name", encoder_name)
        print("Num frames", num_frames)
        print(kwargs)
        if patch_out_dim < 0:
            patch_out_dim = out_dim
        self.use_masked_im_modeling = use_masked_im_modeling
        self.global_ncrops = global_ncrops
        self.local_ncrops = local_ncrops
        self.teacher_3d = teacher_3d
        self.student_3d = student_3d
        self.time_steps = num_frames
        self.model_name = encoder_name
        base_name = '_'.join(['ibot'] + (encoder_name.split('_')[:2]))
        self.student_encoder = EncoderIBOT(model_name=base_name, timm_name=encoder_name, drop_rate=drop_rate,
                                             drop_path_rate=drop_path_rate, pretrained=pretrained, pos_embed=pos_embed,
                                             time_steps=self.time_steps, masked_im_modeling=self.use_masked_im_modeling,
                                             return_all_tokens=True)
        self.student_head = iBOTHead(encoder_embed_dim, out_dim, use_bn=False, norm_last_layer=True, patch_out_dim=patch_out_dim, shared_head=True)
        self.teacher_encoder = EncoderIBOT(model_name=base_name, timm_name=encoder_name, pretrained=pretrained,
                                             pos_embed=pos_embed, time_steps=self.time_steps, return_all_tokens=True)
        self.teacher_head = iBOTHead(encoder_embed_dim, out_dim, use_bn=False, patch_out_dim=patch_out_dim, shared_head=True)
        
        self.student = LatentMAEWrapper(self.student_encoder, self.student_head)
        self.teacher = LatentMAEWrapper(self.teacher_encoder, self.teacher_head)
        self.teacher.load_state_dict(self.student.state_dict(), strict=False)
        for p in self.teacher.parameters():
            p.requires_grad=False
        
        self.loss = LatentMAELoss(out_dim, patch_out_dim,
                            warmup_teacher_temp, teacher_temp,
                            warmup_teacher_patch_temp, teacher_patch_temp,
                            warmup_teacher_temp_epochs, epochs, lambda1=lambda1,
                            lambda2=lambda2, mim_start_epoch=pred_start_epoch,
                            cls_loss=False)

        self.mean = self.student_encoder.mean
        self.std = self.student_encoder.std
        self.num_patches = self.student_encoder.patch_embed.num_patches + 1
        
    # org_x: x without any photometric augmentations
    def forward(self, x, org_x, masks):
        self.student_encoder.model.masked_im_modeling = self.use_masked_im_modeling
        teahcer_x = rearrange(x, 'b c t h w -> (b t) c h w')
        student_x = teahcer_x
        teacher_output = self.teacher(teahcer_x, f2d=False, num_frames=self.time_steps, pool=False)
        student_output = self.student(student_x, mask=masks, f2d=False, num_frames=self.time_steps, pool=False)
        return {'student_output': student_output, 'teacher_output': teacher_output}

    def calculate_loss(self, student_output, teacher_output, masks, epoch):
        loss = self.loss(student_output, teacher_output, masks, epoch, self.student_3d, self.teacher_3d, self.time_steps)
        return loss

    def update_teacher(self, momentum):
        student_param_list = []
        teacher_param_list = []

        with torch.no_grad():
            for ms, mt in zip(self.student.parameters(), self.teacher.parameters()):
                student_param_list.append(ms.data)
                teacher_param_list.append(mt.data)
            torch._foreach_mul_(teacher_param_list, momentum)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1-momentum)     

class LatentMAEWrapper(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x, mask=None, attn_mask=None, num_frames=1, pool=True, f2d=False):
        x = self.encoder(x, mask=mask, attn_mask=attn_mask, num_frames=num_frames, pool=pool, f2d=f2d)
        x = self.head(x)
        return x 

class LatentMAELoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, warmup_teacher_temp, teacher_temp,
                warmup_teacher_patch_temp, teacher_patch_temp,
                wramup_teacher_epochs, nepochs, student_temp=0.1,
                center_momentum=0.9, lambda1=1.0, lambda2=1.0, mim_start_epoch=0, cls_loss=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.cls_loss = cls_loss
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, wramup_teacher_epochs),
            np.ones(nepochs - wramup_teacher_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_patch_temp,
                        teacher_patch_temp, wramup_teacher_epochs),
            np.ones(nepochs - wramup_teacher_epochs) * teacher_patch_temp
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_patch_temp,
            np.linspace(warmup_teacher_patch_temp,
                        teacher_patch_temp, wramup_teacher_epochs),
            np.ones(nepochs - wramup_teacher_epochs - mim_start_epoch) * teacher_patch_temp
        ))

    def forward(self, student_output, teacher_output, student_mask, epoch, student_3d, teacher_3d, num_frames):
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        student_cls = student_cls / self.student_temp
        student_patch = student_patch / self.student_temp
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_patch_temp = self.teacher_temp2_schedule[epoch]
        teacher_cls = F.softmax((teacher_cls - self.center) / teacher_temp, dim=-1).detach()
        teacher_patch = F.softmax((teacher_patch - self.center)/teacher_patch_temp, dim=-1).detach()
        if student_3d and teacher_3d:
            student_patch = rearrange(student_patch, '(b t) n c -> b (t n) c', t=num_frames)
            teacher_patch = rearrange(teacher_patch, '(b t) n c -> b (t n) c', t=num_frames)
            student_mask = rearrange(student_mask, '(b t) h w -> b (t h w)', t=num_frames)
        else:
            student_mask = rearrange(student_mask, '(b t) h w -> (b t) (h w)')
        loss = torch.sum(-teacher_patch * F.log_softmax(student_patch, dim=-1), dim=-1)
        # mask = student_mask.flatten(-2, -1)
        loss = torch.sum(loss * student_mask.float(), dim=-1) / student_mask.sum(dim=-1).clamp(min=1.0)
        loss = loss.mean()
        if self.cls_loss:
            cls_loss = torch.sum(-teacher_cls * F.log_softmax(student_cls, dim=-1), dim=-1)
            cls_loss = cls_loss.mean()
            loss = self.lambda1*loss + self.lambda2*cls_loss
        self.update_center(teacher_cls, teacher_patch)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)  


def latent_mae_vit_small_patch16(**kwargs):
    model = LatentMAE(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        student_3d=True, teacher_3d=True, **kwargs
    )

    return model

def latent_mae_beit_large_patch16(**kwargs):
    model = LatentMAE(
        encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        patch_size=16, encoder_embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        student_3d=True, teacher_3d=True, **kwargs
    )

    return model    
latent_mae_vit_small_patch16 = latent_mae_vit_small_patch16
latent_mae_beit_large_patch16 = latent_mae_beit_large_patch16