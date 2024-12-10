import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from src.models.models_vit import Block
from einops.layers.torch import Rearrange, Reduce

class DecoderViT(nn.Module):
    def __init__(self, num_classes=768, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, camera_params_enabled=False,
                 drop_path_rate=0., attn_drop_rate=0., num_frames=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.camera_params_enabled = camera_params_enabled
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
            norm_layer=norm_layer, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, num_classes, bias=True) # decoder to patch
        
    def forward(self, x, return_token_num, attn_mask=None):
        B, N, C = x.shape 
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)

        if self.camera_params_enabled:
            pc = x.reshape(B*self.num_frames, -1, C)[:, 0, :].reshape(B, self.num_frames, C)
        if return_token_num > 0:
            x = self.decoder_pred(self.decoder_norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.decoder_pred(self.decoder_norm(x))
        if self.camera_params_enabled:
            return x, pc
        return x

class DecoderViT_old(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False, camera_params_enabled=False, num_frames=3):
        super().__init__()
        self.num_classes = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.camera_params_enabled=camera_params_enabled
        self.num_frames = num_frames
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num, attn_mask=None):
        B, N, C = x.shape
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, attn_mask)
        else:   
            for blk in self.blocks:
                x = blk(x, attn_mask)

        if self.camera_params_enabled:
            pc = x.reshape(B*self.num_frames, -1, C)[:, 0, :].reshape(B, self.num_frames, C)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))
        if self.camera_params_enabled:
            return x, pc
        return x