from functools import partial
import timm
import torch
import torch.nn as nn

from einops import rearrange
from src.util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, get_sinusoid_encoding_table
from src.models.backbones import models_encoder
from src.models.backbones.models_decoder import DecoderViT
from src.util.misc import get_cvm_attn_mask
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class VideoMAETimm(nn.Module):
    def __init__(self,
                encoder_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
                backbone_name='EncoderViT',
                patch_size=16,
                encoder_embed_dim=1024,
                decoder_num_classes=768,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=8,
                mlp_ratio=4.,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                init_values=0.,
                use_learnable_pos_emb=False,
                use_checkpoint=False,
                tubelet_size=1,
                pretrained=True,
                camera_params_enabled=False,
                camera_param_dim = 7,
                camera_categories = 64,
                categorical_camera = True,
                num_frames = 4,
                decoder_pos_embed = '1d_spatial',
                decoder_cls = False,
                mask_type = 'causal',
                use_cls = False,
                timm_pool = False,
                patch_embed_type = '2d',
                encoder_3d = False,
                **kwargs
                ):
        super().__init__()
        self.time_steps = num_frames
        self.encoder =  models_encoder.__dict__[backbone_name](model_name=encoder_name, pretrained=pretrained,
                             attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, encoder_3d=encoder_3d,
                            patch_embed_type=patch_embed_type,tubelet_size=tubelet_size, time_steps=self.time_steps)
        self.model_name = encoder_name
        self.mask_type = mask_type
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder = DecoderViT(
            num_classes=decoder_num_classes*tubelet_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            num_frames = self.time_steps,
            camera_params_enabled=camera_params_enabled
            )

        config = timm.data.resolve_model_data_config(self.encoder.model)
        self.mean = config['mean']
        self.std = config['std']
        self.encoder_to_decoder = nn.Sequential(
            nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        )
        self.decoder_pos_embed = decoder_pos_embed  
        self.decoder_cls = decoder_cls

        self.encoder_cls = hasattr(self.encoder.model, 'cls_token') and self.encoder.model.cls_token is not None
        # num_patches = (self.encoder.patch_embed.num_patches + int((decoder_cls and self.encoder_cls)))*self.time_steps
        if encoder_3d:
            time_steps = self.time_steps//tubelet_size
        else:
            time_steps = self.time_steps
        if self.decoder_pos_embed == '1d_spatial':
            self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*time_steps + 0, decoder_embed_dim)
        elif self.decoder_pos_embed == '3d':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches*time_steps + 0, decoder_embed_dim), requires_grad=False)
            grid_size = int(self.encoder.patch_embed.num_patches**0.5)
            self.pos_embed.data.copy_(torch.from_numpy(get_3d_sincos_pos_embed(decoder_embed_dim, (grid_size, grid_size, self.time_steps))).float().unsqueeze(0))

        self.use_cls = use_cls
        self.camera_params_enabled = camera_params_enabled
        self.categorical_camera = categorical_camera
        self.timm_pool = timm_pool
        if self.camera_params_enabled:
            self.camera_encoder = nn.Sequential(
                nn.Linear(camera_param_dim, encoder_embed_dim),
                nn.GELU(),
                nn.LayerNorm(encoder_embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(encoder_embed_dim, encoder_embed_dim),
                nn.GELU(),
                nn.LayerNorm(encoder_embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(encoder_embed_dim, decoder_embed_dim)
            )
            if self.categorical_camera:
                self.camera_decoder = nn.Sequential(
                    nn.Linear(decoder_embed_dim, camera_categories)
                )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward_masked(self, x, mask, camera):
        B_orig, _, T, _, _ = x.shape
        # x = rearrange(x, 'b c t h w -> (b t) c h w')
        encoder_features = self.encoder(x, mask, f2d=False) # [B, N_vis, C_e]
        x = self.encoder_to_decoder(encoder_features) # [B, N_vis, C_d]
        if self.timm_pool:
            encoder_features = self.encoder.model.forward_head(encoder_features, pre_logits=True)
        else:
            encoder_features = torch.mean(encoder_features, dim=1)
        if self.encoder_cls and not self.decoder_cls:
            x_cls_token = x[:, :1, :]
            x = x[:, 1:, :]
        # B, N, C = x.shape
        # x = x.reshape(B_orig, -1, C)
        x = rearrange(x, '(b t) n c -> b (t n) c', b=B_orig)
        B, N, C = x.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)

        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x = torch.cat([x + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        
        if self.camera_params_enabled:
            pred_cams = x[1]
            x = x[0]
        if self.camera_params_enabled:
            if self.categorical_camera:
                pred_cams = self.camera_decoder(pred_cams)

        output = {
            'pred_camera': (pred_cams if self.camera_params_enabled else None),
            'gt_camera': (gt_embed if self.camera_params_enabled else None),
            'encoder_features': encoder_features,
            'pred_frames': x,   
        }
        return output

    def forward(self, x, mask, camera=None):
        return self.forward_masked(x, mask, camera=camera)

def videomae_vit_small_patch16_dec192d4(**kwargs):
    model = VideoMAETimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def autoencoder_vit_small_patch16_dec192d4(**kwargs):
    model = VideoMAETimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def videomae_3d_3d_vit_small_2dpatch16_dec192d4(**kwargs):
    model = VideoMAETimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), tubelet_size=2,
        encoder_3d=True, patch_embed_type='2d', **kwargs)
    return model    

def videomae_beit_large_patch16_dec512d8(**kwargs):
    model = VideoMAETimm(encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        backbone_name='EncoderBeit', patch_size=16, encoder_embed_dim=1024, depth=24, 
        num_heads=16,decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def videomae_3d_3d_beit_large_patch16_dec512d8(**kwargs):
    model = VideoMAETimm(encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        backbone_name='EncoderBeit', patch_size=16, encoder_embed_dim=1024, depth=24, 
        num_heads=16,decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), encoder_3d=True, patch_embed_type='2d', **kwargs)
    return model 

videomae_vit_small_patch16 = videomae_vit_small_patch16_dec192d4 # decoder: 192 dim, 4 blocks
videomae_beit_large_patch16 = videomae_beit_large_patch16_dec512d8
autoencoder_vit_small_patch16 = autoencoder_vit_small_patch16_dec192d4

videomae_3d_vit_small_2dpatch16 = videomae_3d_3d_vit_small_2dpatch16_dec192d4 # decoder: 192 dim, 4 blocks
videomae_3d_beit_large_2dpatch16 = videomae_3d_3d_beit_large_patch16_dec512d8