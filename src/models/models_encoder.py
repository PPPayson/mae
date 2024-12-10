from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from src.util.pos_embed import get_2d_sincos_pos_embed
import timm

class EncoderViT(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, drop_rate=0, drop_path_rate=0, **kwargs):
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

    def forward_encoder(self, x, mask):

        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = x.movedim(1, 2).reshape(B*T, C, H, W) # 2d encoder process frames separately   
        x = self.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        B, _, C = x.shape #B = B*T | C --> enc emd dim
        selector = ~mask.view(B, -1)
        if (~selector[0]).sum() == 0:  #causal mode
            B = int(B*0.75)
        x = x[selector].reshape(B, -1, C)
        # append cls token
        cls_token = cls_token[:, None, :]
        x = torch.cat((cls_token, x), dim=1)
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

    def initialize_weights(self):
        # initialization
        if not self.pretrained:
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.cls_token, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.7):
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

class FullModel(nn.Module):
    def __init__(self, encoder, head, pool=False):
        super(FullModel, self).__init__()
        self.head = head
        self.encoder = encoder
        self.pool = pool
    def forward(self, x):
        x = self.encoder(x, f2d=True, pool=self.pool)
        if len(x.shape) > 2:
            x = x.mean(dim=1)
        return self.head(x)