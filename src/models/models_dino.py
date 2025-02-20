from functools import partial
from itertools import chain
import timm
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from src.models.models_encoder import EncoderViT, EncoderDINO
from src.models.models_decoder import DecoderViT
from src.util.dino import trunc_normal_, cancel_gradients_last_layer, cosine_scheduler
from src.util.misc import get_cvm_attn_mask

class DinoTimm(nn.Module):
    def __init__(self,
                encoder_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
                encoder_embed_dim=1024,
                out_dim=65536,
                warmup_teacher_temp=0.04,
                teacher_temp=0.04,
                warmup_teacher_temp_epochs=10,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                tubelet_size=1,
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
                **kwargs
                ):
        super().__init__()
        print("Pretrained", pretrained)
        print('Pos type', pos_embed)
        print("Encoder name", encoder_name)
        print("Num frames", num_frames)
        print(kwargs)
        self.teacher_3d = teacher_3d
        self.student_3d = student_3d
        self.time_steps = num_frames-1
        self.model_name = encoder_name
        base_name = '_'.join(encoder_name.split('_')[:2])
        self.student_encoder = EncoderDINO(model_name=base_name, timm_name=encoder_name, drop_rate=drop_rate, drop_path_rate=drop_path_rate, pretrained=pretrained, pos_embed=pos_embed, time_steps=self.time_steps)
        self.student_head = DINOHead(encoder_embed_dim, out_dim, use_bn=False, norm_last_layer=True)
        self.teacher_encoder = EncoderDINO(model_name=base_name, timm_name=encoder_name, pretrained=pretrained, pos_embed=pos_embed, time_steps=self.time_steps)
        self.teacher_head = DINOHead(encoder_embed_dim, out_dim, use_bn=False)
        
        self.student = DINOWrapper(self.student_encoder, self.student_head)
        self.teacher = DINOWrapper(self.teacher_encoder, self.teacher_head)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad=False
        
        self.dino_loss = DINOLoss(out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, epochs)

        self.mean = self.student_encoder.mean
        self.std = self.student_encoder.std
        self.num_patches = self.student_encoder.patch_embed.num_patches + 1

    def forward(self, x):
        if self.teacher_3d:
            teacher_x = x[:, :, 1:, :, :]
            teacher_x = rearrange(teacher_x, 'b c t h w -> (b t) c h w')
            teacher_attn_mask = get_cvm_attn_mask(self.num_patches*(self.time_steps), self.time_steps, offset=1).to(x.device)
            teacher_output = self.teacher(teacher_x, teacher_attn_mask, pool=True, f2d=False, num_frames=self.time_steps)
        elif self.student_3d:
            teacher_x = x[:, :, 1:, :, :]
            teacher_x = rearrange(teacher_x, 'b c t h w -> (b t) c h w')
            teacher_output = self.teacher(teacher_x, pool=True, f2d=True, num_frames=self.time_steps)
        else:
            teacher_x = x[: ,:, -1, :, :]
            teacher_output = self.teacher(teacher_x, pool=True, f2d=True, num_frames=1)

        student_x = x[:, :, :self.time_steps, :, :] # Leave out last frame for prediction only
        student_x = rearrange(student_x, 'b c t h w -> (b t) c h w')

        if self.student_3d:
            student_attn_mask = get_cvm_attn_mask(self.num_patches*(self.time_steps), self.time_steps).to(x.device)
            student_output = self.student(student_x, student_attn_mask, num_frames=self.time_steps, pool=True, f2d=False)
        else:
            student_output = self.student(student_x, num_frames=self.time_steps, pool=True, f2d=True)
        return {'student_output': student_output, 'teacher_output': teacher_output}

    def calculate_loss(self, student_output, teacher_output, epoch):
        student_output = rearrange(student_output, '(b t) c -> b t c', t=self.time_steps)
        if self.student_3d:
            student_output = student_output[:, 1:]
        decoder_time_steps = self.time_steps if (self.teacher_3d or self.student_3d) else 1
        teacher_output = rearrange(teacher_output, '(b t) c -> b t c', t=decoder_time_steps)
        # if not self.teacher_3d and self.student_3d:
        #     teacher_output = torch.repeat_interleave(teacher_output, dim=1, repeats=self.time_steps-1, output_size=self.time_steps-1)
        if not self.student_3d and not self.teacher_3d:
            teacher_output = torch.repeat_interleave(teacher_output, dim=1, repeats=self.time_steps, output_size=self.time_steps)
        else:
            teacher_output = teacher_output[:, 1:]
        student_output = rearrange(student_output, 'b t c -> (b t) c')
        teacher_output = rearrange(teacher_output, 'b t c -> (b t) c')
        loss = self.dino_loss(student_output, teacher_output, epoch)
        
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



# class AutoregDinoTimm(nn.Module):
#     def __init__(self,
#                 encoder_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
#                 encoder_embed_dim=1024,
#                 out_dim=65536,
#                 warmup_teacher_temp=0.04,
#                 teacher_temp=0.04,
#                 warmup_teacher_temp_epochs=10,
#                 drop_rate=0.,
#                 attn_drop_rate=0.,
#                 drop_path_rate=0.,
#                 norm_layer=nn.LayerNorm,
#                 tubelet_size=1,
#                 pretrained=False,
#                 camera_params_enabled=False,
#                 camera_param_dim = 7,
#                 camera_categories = 64,
#                 categorical_camera = True,
#                 num_frames = 4,
#                 decoder_cls = False,
#                 timm_pool = False,
#                 epochs = 30,
#                 pos_embed='3d',
#                 **kwargs
#                 ):
#         super().__init__()
#         self.time_steps = num_frames-1
#         self.model_name = encoder_name
#         base_name = '_'.join(encoder_name.split('_')[:2])
#         self.student_encoder = EncoderDINO(model_name=base_name, timm_name=encoder_name, drop_rate=drop_rate, drop_path_rate=drop_path_rate, pretrained=pretrained, time_steps=self.time_steps, pos_embed=pos_embed)
#         self.student_head = DINOHead(encoder_embed_dim, out_dim, use_bn=False, norm_last_layer=True)
#         self.teacher_encoder = EncoderDINO(model_name=base_name, timm_name=encoder_name, pretrained=pretrained, time_steps=self.time_steps, pos_embed=pos_embed)
#         self.teacher_head = DINOHead(encoder_embed_dim, out_dim, use_bn=False)
        
#         self.student = DINOWrapper(self.student_encoder, self.student_head)
#         self.teacher = DINOWrapper(self.teacher_encoder, self.teacher_head)
#         self.teacher.load_state_dict(self.student.state_dict())
#         for p in self.teacher.parameters():
#             p.requires_grad=False
        
#         self.dino_loss = DINOLoss(out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, epochs)

#         self.mean = self.student_encoder.mean
#         self.std = self.student_encoder.std
#         self.num_patches = self.student_encoder.patch_embed.num_patches + 1

#     def forward(self, x):
#         student_x = x[:, :, :self.time_steps, :, :] # Leave out last frame for prediction only
#         teacher_x = x[:, :, 1:, :, :]
#         attn_mask = get_cvm_attn_mask(self.num_patches*(self.time_steps), self.time_steps).to(x.device)
#         teacher_attn_mask = get_cvm_attn_mask(self.num_patches*(self.time_steps), self.time_steps, offset=1).to(x.device)
#         student_x = rearrange(student_x, 'b c t h w -> (b t) c h w')
#         teacher_x = rearrange(teacher_x, 'b c t h w -> (b t) c h w')

#         student_output = self.student(student_x, attn_mask, num_frames=self.time_steps, pool=True, f2d=False)
#         teacher_output = self.teacher(teacher_x, attn_mask=teacher_attn_mask, pool=True, f2d=False, num_frames=self.time_steps)
#         return {'student_output': student_output, 'teacher_output': teacher_output}

#     def calculate_loss(self, student_output, teacher_output, epoch):
#         student_output = rearrange(student_output, '(b t) c -> b t c', t=self.time_steps)[:, 1:]
#         teacher_output = rearrange(teacher_output, '(b t) c -> b t c', t=self.time_steps)[:, 1:]
#         student_output = rearrange(student_output, 'b t c -> (b t) c')
#         teacher_output = rearrange(teacher_output, 'b t c -> (b t) c')
#         loss = self.dino_loss(student_output, teacher_output, epoch)
        
#         return loss

#     def update_teacher(self, momentum):
#         student_param_list = []
#         teacher_param_list = []
#         with torch.no_grad():
#             for ms, mt in zip(self.student.parameters(), self.teacher.parameters()):
#                 student_param_list.append(ms.data)
#                 teacher_param_list.append(mt.data)
#             # for k in self.student.named_parameters().keys():
#             #     for ms, mt in zip(self.student[k], self.teacher[k]):
#             #         student_param_list += ms.params
#             #         teacher_param_list += mt.params
#             torch._foreach_mul_(teacher_param_list, momentum)
#             torch._foreach_add_(teacher_param_list, student_param_list, alpha=1-momentum)

# Original implementation https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L257
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOWrapper(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x, attn_mask=None, num_frames=1, pool=True, f2d=False):
        x = self.encoder(x, attn_mask=attn_mask, num_frames=num_frames, pool=pool, f2d=f2d)
        x = self.head(x)
        return x 
        
class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        # student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach()
        #teacher_out = teacher_out.detach().chunk(2)
        total_loss = F.cross_entropy(input=student_out, target=teacher_out)
        #total_loss = 0
        #n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # we skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        #total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def autoreg_dino_vit_small_patch16(**kwargs):
    model = DinoTimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), student_3d=True, teacher_3d=False, **kwargs)
    return model 

def dino_2d_vit_small_patch16(**kwargs):
    model = DinoTimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), student_3d=False, teacher_3d=False, **kwargs)
    return model 

def dino_3d_vit_small_patch16(**kwargs):
    model = DinoTimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        student_3d=True, teacher_3d=True, **kwargs)
    return model 

def autoreg_dino_beit_large_patch16(**kwargs):
    print(kwargs)
    model = DinoTimm(
        encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        patch_size=16, encoder_embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        student_3d=True, teacher_3d=False, **kwargs
    )
    return model
    
def dino_2d_beit_large_patch16(**kwargs):
    model = DinoTimm(
        encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        patch_size=16, encoder_embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        student_3d=True, teacher_3d=False, **kwargs
    )
    return model

def dino_3d_beit_large_patch16(**kwargs):
    model = DinoTimm(
        encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        patch_size=16, encoder_embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        student_3d=True, teacher_3d=False, **kwargs
    )
    return model

autoreg_dino_vit_small_patch16 = autoreg_dino_vit_small_patch16
dino_2d_vit_small_patch16 = dino_2d_vit_small_patch16
dino_3d_vit_small_patch16 = dino_3d_vit_small_patch16

autoreg_dino_beit_large_patch16 = autoreg_dino_beit_large_patch16
dino_2d_beit_large_patch16 = dino_2d_beit_large_patch16
dino_3d_beit_large_patch16 = dino_3d_beit_large_patch16