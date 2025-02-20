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
                 drop_path_rate=0, patch_embed_type='2d', tubelet_size=2, encoder_3d=False, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
            )
        self.pretrained=pretrained
        self.model_name = model_name
        print("PATCH", patch_embed_type)
        if patch_embed_type == '2d':
            self.patch_embed = self.model.patch_embed
        else:
            img_size = self.model.patch_embed.img_size
            patch_size = self.model.patch_embed.patch_size
            embed_dim = self.model.embed_dim
            self.patch_embed = PatchEmbed3D(img_size, patch_size, tubelet_size=tubelet_size, embed_dim=embed_dim)
            self.model.patch_embed = self.patch_embed
        self.embed_dim = self.model.num_features
        self.encoder_3d = encoder_3d
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward_encoder(self, x, mask):
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            if not self.encoder_3d:
                x = x.movedim(1, 2).reshape(B*T, C, H, W) # 2d encoder process frames separately 
        x = self.patch_embed(x) # if 3d: B t//2*N C else B*T N C
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        # cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        B, _, C = x.shape #B = B*T | C --> enc emd dim
        selector = ~mask.view(B, -1)
        if (~selector[0]).sum() == 0:  #causal mode
            B = int(B*0.75)
        x = x[selector].reshape(B, -1, C)
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
    def __init__(self, model_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k', pretrained=True, drop_rate=0, drop_path_rate=0, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
            )
        self.pretrained=pretrained
        self.model_name = model_name
        self.patch_embed = self.model.patch_embed
        self.embed_dim = self.model.num_features
        self.cls_token = self.model.cls_token
        # Remove unused norm for ddp, we add back the norm in linear head 
        self.model.fc_norm = nn.Identity()
    def forward_encoder(self, x, mask):
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            # x = x.movedim(1, 2).reshape(B*T, C, H, W) # 2d encoder process frames separately   
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.model.pos_embed is not None:
            x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        rel_pos_bias = self.model.rel_pos_bias() if self.model.rel_pos_bias is not None else None

        cls_tokens = x[:, :1, :]
        x = x[:, 1:, :]
        B, _, C = x.shape #B = B*T | C --> enc emd dim
        selector = ~mask.view(B, -1)
        if (~selector[0]).sum() == 0:  #causal mode
            B = int(B*0.75)
        x = x[selector].reshape(B, -1, C)
        # append cls token
        # cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.model.blocks:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)
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

class EncoderVolo(nn.Module):
    def __init__(self, model_name='volo_d1_224.sail_in1k', pretrained=True, drop_rate=0, drop_path_rate=0, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
            )
        self.pretrained=pretrained
        self.model_name = model_name
        self.patch_embed = self.model.patch_embed
        self.embed_dim = self.model.num_features
        self.cls_token = self.model.cls_token
        # Remove unused norm for ddp, we add back the norm in linear head 
        del self.model.fc_norm
    def forward_encoder(self, x, mask):
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = x.movedim(1, 2).reshape(B*T, C, H, W) # 2d encoder process frames separately   
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        cls_tokens = x[:, 0, :]
        x = x[:, 1:, :]
        B, _, C = x.shape #B = B*T | C --> enc emd dim
        selector = ~mask.view(B, -1)
        if (~selector[0]).sum() == 0:  #causal mode
            B = int(B*0.75)
        x = x[selector].reshape(B, -1, C)
        # append cls token
        # cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.norm(x)        
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
