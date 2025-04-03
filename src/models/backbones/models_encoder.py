from functools import partial
import timm
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from src.models import models_dict
from src.util.pos_embed import get_2d_sincos_pos_embed
from src.util.pos_embed import get_3d_sincos_pos_embed
from einops import rearrange
from src.util.misc import PatchEmbed3D
from src.data.masking_generator import mask2attn_mask

class EncoderIBOT(nn.Module):
    def __init__(self, model_name='ibot_vit_small', timm_name='vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=False, drop_rate=0,
                         drop_path_rate=0, pos_embed='2d', time_steps=1, masked_im_modeling=False, return_all_tokens=False, **kwargs):
        super().__init__()
        self.model = models_dict.__dict__[model_name](num_classes=0, drop_rate=drop_rate, 
                                                drop_path_rate=drop_path_rate, pos_embed=pos_embed,
                                                time_steps=time_steps, masked_im_modeling=masked_im_modeling,
                                                return_all_tokens=return_all_tokens)
        print(model_name, timm_name, pretrained)
        if pretrained:
            assert timm_name is not None
            timm_model = timm.create_model(timm_name, pretrained=True, num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
            msg = self.model.load_state_dict(timm_model.state_dict(), strict=False)
            print(msg)
            config = timm.data.resolve_model_data_config(timm_model)
            self.mean = config['mean']
            self.std = config['std']
            del timm_model
        else:
            self.mean = [.5, .5, .5]
            self.std = [.5, .5, .5]
        self.time_steps = time_steps
        if pos_embed == '3d':
            self.temporal_embed = nn.Parameter(torch.zeros(1, self.time_steps, 1, self.model.embed_dim))
        else:
            self.temporal_embed = None
        self.pretrained=pretrained
        self.model_name=model_name
        self.timm_name=timm_name
        self.patch_embed=self.model.patch_embed
        self.embed_dim=self.model.num_features
    def forward(self, x, attn_mask=None, num_frames=1, pool=False, f2d=False, mask=None):
        x = self.model(x, attn_mask=attn_mask, num_frames=num_frames, pool=pool, f2d=f2d, temporal_embed=self.temporal_embed, mask=mask)
        return x

# ViT encoder with support for attn masks
class EncoderDINO(nn.Module):
    def __init__(self, model_name='vit_small', timm_name='vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=False, drop_rate=0,
                         drop_path_rate=0, pos_embed='2d', time_steps=1, **kwargs):
        super().__init__()
        self.model = models_dict.__dict__[model_name](num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate, pos_embed=pos_embed, time_steps=time_steps)
        print(model_name, timm_name, pretrained)
        if pretrained:
            assert timm_name is not None
            timm_model = timm.create_model(timm_name, pretrained=True, num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
            msg = self.model.load_state_dict(timm_model.state_dict(), strict=(pos_embed=='2d'))
            print(msg)
            config = timm.data.resolve_model_data_config(timm_model)
            self.mean = config['mean']
            self.std = config['std']
            del timm_model
        else:
            self.mean = [.5, .5, .5]
            self.std = [.5, .5, .5]
        self.time_steps = time_steps
        if pos_embed == '3d':
            self.temporal_embed = nn.Parameter(torch.zeros(1, self.time_steps, 1, self.model.embed_dim))
        else:
            self.temporal_embed = None
        self.pretrained=pretrained
        self.model_name=model_name
        self.timm_name=timm_name
        self.patch_embed=self.model.patch_embed
        self.embed_dim=self.model.num_features
    def forward(self, x, attn_mask=None, num_frames=1, pool=False, f2d=False):
        x = self.model(x, attn_mask, num_frames, pool, f2d=f2d, temporal_embed=self.temporal_embed)
        return x
        

class EncoderViT(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, drop_rate=0,
                 drop_path_rate=0, encoder_3d=False, pos_embed='2d', time_steps=1, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            dynamic_img_size=True
            )
        self.time_steps = time_steps
        self.pretrained=pretrained
        self.model_name = model_name
        self.patch_embed = self.model.patch_embed
        # if patch_embed_type == '2d':
        #     self.patch_embed = self.model.patch_embed
        # else:
        #     img_size = self.model.patch_embed.img_size
        #     patch_size = self.model.patch_embed.patch_size
        #     embed_dim = self.model.embed_dim
        #     self.patch_embed = PatchEmbed3D(img_size, patch_size, tubelet_size=tubelet_size, embed_dim=embed_dim)
        #     self.model.patch_embed = self.patch_embed
        self.embed_dim = self.model.num_features
        self.encoder_3d = encoder_3d
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward_encoder(self, x, mask):
        B = x.shape[0]
        if len(x.shape) == 5:
            # B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x) # if 3d: B t//2*N C else B*T N C

        # if not self.encoder_3d:
        x = self.model._pos_embed(x)
        if self.encoder_3d:
            x = rearrange(x, '(b t) n c -> b (t n) c', b=B)
        # else:
        #     pos_embed = self.model.pos_embed
        #     cls_pos_embed = pos_embed[:, :1, :]
        #     pos_embed = pos_embed[:, 1:, :].repeat(1, int(x.shape[1]/(pos_embed.shape[1]-1)), 1)
        #     pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)
        #     to_cat = []
        #     if self.model.cls_token is not None:
        #         to_cat.append(self.model.cls_token.expand(x.shape[0], -1, -1))
        #     if self.model.reg_token is not None:
        #         to_cat.append(self.model.reg_token.expand(x.shape[0], -1, -1))
        #     x = torch.cat(to_cat + [x], dim=1)
        #     x = x + pos_embed
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        if self.encoder_3d:
            x = rearrange(x, 'b (t n) c -> (b t) n c', t=self.time_steps)
        # cls_token = x[:, 0, :]

        x = x[:, 1:, :]
        if self.encoder_3d:
            x = rearrange(x, '(b t) n c -> b (t n) c', b=B)
        B, _, C = x.shape #B = B*T | C --> enc emd dim
        selector = ~mask.view(B, -1)
        if (~selector[0]).sum() == 0 and (~selector).sum() != 0:  #causal mode
            B = int(B*0.75)
        x = x[selector]
        x = x.reshape(B, -1, C)
        # x = x[selector].reshape(B, -1, C)
        # append cls token
        cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x

    def forward_2D(self, x, pool=False):
        x = self.model.forward_features(x)
        if pool:
            x = self.model.forward_head(x, pre_logits=True)
        return x

    def forward(self, imgs, mask=None, f2d=False, pool=False):
        if f2d:
            latent = self.forward_2D(imgs, pool)
        else:
            latent = self.forward_encoder(imgs, mask)
        return latent

class EncoderBeit(nn.Module):
    def __init__(self, model_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k', pretrained=True, drop_rate=0,
                     drop_path_rate=0, encoder_3d=False, pos_embed='2d', time_steps=1, **kwargs):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name,
        #     pretrained=pretrained,
        #     num_classes=0,
        #     drop_rate=drop_rate,
        #     drop_path_rate=drop_path_rate
        #     )
        self.encoder_3d = encoder_3d
        # if self.encoder_3d:
        time_steps = time_steps
        # else:
            # time_steps = 1
        self.time_steps = time_steps  
        timm_name = model_name
        model_name = '_'.join(model_name.split('_')[:2])
        self.model = models_dict.__dict__[model_name](num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate, pos_embed=pos_embed, time_steps=time_steps)
        
        if pretrained:
            timm_model = timm.create_model(timm_name, pretrained=True, num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
            msg = self.model.load_state_dict(timm_model.state_dict(), strict=True)
            print(msg)
            config = timm.data.resolve_model_data_config(timm_model)
            self.mean = config['mean']
            self.std = config['std']
            del timm_model
        else:
            self.mean = [.5, .5, .5]
            self.std = [.5, .5, .5]
        self.pretrained=pretrained
        self.model_name = timm_name
        self.patch_embed = self.model.patch_embed
        self.embed_dim = self.model.num_features
        self.cls_token = self.model.cls_token
        # Remove unused norm for ddp, we add back the norm in linear head 
        self.model.fc_norm = nn.Identity()

    def forward_encoder(self, x, mask):
        attn_mask, mask = mask2attn_mask(~mask, f2d=(not self.encoder_3d), num_frames=self.time_steps)
        attn_mask = attn_mask[:, None, :, :]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.model(x, attn_mask=attn_mask, f2d=(not self.encoder_3d), pool=False)
        B, N, C = x.shape
        selector = mask.view(B, -1)
        if (~selector[0]).sum() == 0:  #causal mode
            B = int(B*0.75)
        x = x[selector].reshape(B, -1, C)               
        return x

    def forward_2D(self, x, pool=False):
        x = self.model.forward_features(x, pool=False, f2d=True)
        if pool:
            x = self.model.forward_head(x, pre_logits=True)
        return x

    def forward(self, imgs, mask=None, f2d=False, pool=False):
        if f2d:
            latent = self.forward_2D(imgs, pool)
        else:
            latent = self.forward_encoder(imgs, mask)
        return latent


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.linear = torch.nn.Linear(input_dim, num_classes)
        #self.norm = torch.nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.norm = torch.nn.LayerNorm(input_dim)
    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class LinearModelTimm(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.7):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.linear = torch.nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x    

class FullModel(nn.Module):
    def __init__(self, encoder, head, pool=False, timm_model=False):
        super(FullModel, self).__init__()
        self.head = head
        self.encoder = encoder
        self.pool = pool
        self.timm_model = timm_model
    def forward(self, x):
        if not self.timm_model:
            x = self.encoder(x, f2d=True, pool=self.pool)
        else:
            x = self.encoder.forward_head(self.encoder.forward_features(x), pre_logits=True)
        if len(x.shape) > 2:
            x = x.mean(dim=1)
        return self.head(x)
