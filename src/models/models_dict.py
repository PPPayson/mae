from src.models.backbones.models_beit import Beit
from src.models.backbones.models_dino_vit import VisionTransformer
from src.models.backbones.models_ibot_vit import VisionTransformer as IBOTVisionTransformer
from functools import partial
import torch.nn as nn

def beitv2_large(patch_size=16, **kwargs):
    model = Beit(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, use_abs_pos_emb=False,
        use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    return model

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ibot_vit_small(patch_size=16, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs        
        )
    return model

def ibot_beitv2_large(patch_size=16, **kwargs):
    model = Beit(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, use_abs_pos_emb=False,
        use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    return model
