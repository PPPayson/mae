import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from src.models.backbones.models_vit import Block
from einops.layers.torch import Rearrange, Reduce

class DecoderWrapper(nn.Module):
    def __init__(self, encoder_to_decoder, decoder):
        super().__init__()
        self.encoder_to_decoder = encoder_to_decoder
        self.decoder = decoder

    def forward(self, x, return_token_num, attn_mask=None):
        return self.decoder(self.encoder_to_decoder(x), return_token_num=return_token_num, attn_mask=attn_mask)
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
    
