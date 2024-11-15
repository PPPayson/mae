from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from src.util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.models.models_2D import Masked2DEncoderTIMM

class AutoregViTtimm(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # ---------------------------------------------------------------------
        # Autoreg TIMM encoder


