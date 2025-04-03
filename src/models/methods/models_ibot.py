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
from src.models.methods.models_dino import DINOHead
from src.models.backbones.models_dino_vit import MultiCropWrapper

class IbotTimm(nn.Module):
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
        self.student_head = iBOTHead(encoder_embed_dim, out_dim, use_bn=False, norm_last_layer=True, patch_out_dim=patch_out_dim)
        self.teacher_encoder = EncoderIBOT(model_name=base_name, timm_name=encoder_name, pretrained=pretrained,
                                             pos_embed=pos_embed, time_steps=self.time_steps, return_all_tokens=True)
        self.teacher_head = iBOTHead(encoder_embed_dim, out_dim, use_bn=False, patch_out_dim=patch_out_dim)
        
        self.student = MultiCropWrapper(self.student_encoder, self.student_head)
        self.teacher = MultiCropWrapper(self.teacher_encoder, self.teacher_head)
        self.teacher.load_state_dict(self.student.state_dict(), strict=False)
        for p in self.teacher.parameters():
            p.requires_grad=False
        
        self.ibot_loss = IBOTLoss(out_dim, patch_out_dim, global_ncrops,
                            local_ncrops, warmup_teacher_temp, teacher_temp,
                            warmup_teacher_patch_temp, teacher_patch_temp,
                            warmup_teacher_temp_epochs, epochs, lambda1=lambda1,
                            lambda2=lambda2, mim_start_epoch=pred_start_epoch)

        self.mean = self.student_encoder.mean
        self.std = self.student_encoder.std
        self.num_patches = self.student_encoder.patch_embed.num_patches + 1
        
    # org_x: x without any photometric augmentations
    def forward(self, x, org_x, masks):
        self.student.backbone.model.masked_im_modeling = self.use_masked_im_modeling
        teacher_x = x[:self.global_ncrops]
        teacher_x = [rearrange(x_i, 'b c t h w -> (b t) c h w') for x_i in teacher_x]
        student_x = teacher_x
        # teacher_x = rearrange(teacher_x, 'n b c t h w -> (b n t) c h w')
        # student_x = rearrange(student_x, 'n b c t h w -> (b n t) c h w')
        teacher_output = self.teacher(teacher_x, f2d=True, num_frames=self.time_steps, pool=False)
        student_output = self.student(student_x, mask=masks[:self.global_ncrops], f2d=True, num_frames=self.time_steps, pool=False)
        student_x = x[self.global_ncrops:]

        student_x = [rearrange(x_i, 'b c t h w -> (b t) c h w') for x_i in student_x]
        self.student.backbone.model.masked_im_modeling = False
        student_local_cls = self.student(student_x)[0] if len(x) > self.global_ncrops else None
        self.student.backbone.model.masked_im_modeling = self.use_masked_im_modeling
        return {'student_output': [student_output, student_local_cls], 'teacher_output': teacher_output}

    def calculate_loss(self, student_output, teacher_output, masks, epoch):
        student_local_cls = student_output[1]
        student_output = student_output[0]
        all_loss = self.ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
        loss = all_loss.pop('loss')
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

class IBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_patch_temp, teacher_patch_temp, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_patch_temp,
                        teacher_patch_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_patch_temp
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_patch_temp,
            np.linspace(warmup_teacher_patch_temp,
                        teacher_patch_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_patch_temp
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

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


class iBOTHead(DINOHead):

    def __init__(self, *args, patch_out_dim=8192, norm=None, #act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, **kwargs):
        
        super(iBOTHead, self).__init__(*args,
                                        # norm=norm,
                                        # act=act,
                                        # last_norm=last_norm,
                                        nlayers=nlayers,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None
            #TODO Add back norm
            # self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            # self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super().forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        
       #  if self.last_norm is not None:
        #     x1 = self.last_norm(x1)
        #     x2 = self.last_norm2(x2)
        
        return x1, x2

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def ibot_2d_vit_small_patch16(**kwargs):
    print(kwargs)
    model = IbotTimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def ibot_3d_vit_small_patch16(**kwargs):
    print(kwargs)
    model = IbotTimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        teacher_3d=True,  student_3d=True, **kwargs
    )
    return model

def ibot_2d_beit_large_patch16(**kwargs):
    print(kwargs)
    model = IbotTimm(
        encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        patch_size=16, encoder_embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        teacher_3d=False,  student_3d=False, **kwargs
    )
    return model

ibot_2d_vit_small_patch16 = ibot_2d_vit_small_patch16
ibot_3d_vit_small_patch16 = ibot_3d_vit_small_patch16
ibot_2d_beit_large_patch16 = ibot_2d_beit_large_patch16