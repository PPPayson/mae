from functools import partial
import timm
import torch
import torch.nn as nn

from einops import rearrange
from src.util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
# from src.models.models_encoder import EncoderViT
from src.models.backbones.models_decoder import DecoderViT
from src.util.misc import get_cvm_attn_mask
from src.models.backbones import models_encoder

class AutoregViTtimm(nn.Module):
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
                decoder_pos_embed = '3d',
                decoder_cls = False,
                timm_pool = False,
                **kwargs
                ):
        super().__init__()
        # ---------------------------------------------------------------------
        # Autoreg TIMM encoder
        # self.encoder = EncoderViT(model_name=encoder_name, drop_rate=drop_rate, drop_path_rate=drop_path_rate, pretrained=pretrained)
        self.encoder = models_encoder.__dict__[backbone_name](model_name=encoder_name, drop_rate=drop_rate, drop_path_rate=drop_path_rate, pretrained=pretrained)
        self.model_name = encoder_name
        self.time_steps = num_frames-1
        self.timm_pool = timm_pool
        self.decoder_cls = decoder_cls
        self.decoder = DecoderViT(
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            camera_params_enabled=camera_params_enabled,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            num_frames=self.time_steps            
        )

        config = timm.data.resolve_model_data_config(self.encoder.model)
        self.mean = config['mean']
        self.std = config['std']
        self.encoder_to_decoder = nn.Sequential(
            nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        )

        self.encoder_cls = hasattr(self.encoder.model, 'cls_token') and self.encoder.model.cls_token is not None
        # num_patches = (self.encoder.patch_embed.num_patches + int((decoder_cls and self.encoder_cls)))*self.time_steps
        num_patches = (self.encoder.patch_embed.num_patches)*self.time_steps
        self.decoder_pos_embed = decoder_pos_embed
        if decoder_pos_embed == '2d':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
            grid_size = int(self.encoder.patch_embed.num_patches**0.5)
            self.pos_embed.data.copy_(torch.from_numpy(get_2d_sincos_pos_embed(decoder_embed_dim, grid_size)).float().unsqueeze(0))
        elif decoder_pos_embed == '3d':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
            grid_size = int(self.encoder.patch_embed.num_patches**0.5)
            self.pos_embed.data.copy_(torch.from_numpy(get_3d_sincos_pos_embed(decoder_embed_dim, (grid_size, grid_size, self.time_steps))).float().unsqueeze(0))            
        elif decoder_pos_embed == 'learned_3d':
            self.temporal_embed = nn.Parameter(torch.zeros(1, self.time_steps, 1, decoder_embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches, decoder_embed_dim))

        self.camera_params_enabled = camera_params_enabled
        self.categorical_camera = categorical_camera

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
        return {'pos_embed', 'cls_token', 'mask_token', 'temporal_embed'}

    def forward(self, x, camera=None, mask=None):
        x = x[:, :, :self.time_steps, :, :] # Leave out last frame for prediction only
        batch_size = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w') # Merge timestep into batch size for encoding
        encoder_features = self.encoder(x, f2d=True)
        x = self.encoder_to_decoder(encoder_features)
        if self.timm_pool:
            # Use pooling predefined in timm
            encoder_features = self.encoder.model.forward_head(encoder_features, pre_logits=True)
        else:
            # Global average pooling
            encoder_features = torch.mean(encoder_features, dim=1)
        
        if self.encoder_cls:
            x_cls_token = x[:, :1, :]
            x = x [:, 1:, :]

        if self.decoder_pos_embed == 'learned_3d':
            B, N, C = x.shape
            expand_temporal_embed = self.temporal_embed.expand(batch_size, -1, -1, -1).type_as(x).to(x.device)
            expand_temporal_embed = rearrange(expand_temporal_embed, 'b t n c -> (b t) n c')
            expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device)
            x = x + self.pos_embed + expand_temporal_embed
        else:
            x = rearrange(x, '(b t) n c -> b (t n) c', t=self.time_steps)
            B, N, C = x.shape
            expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device)
            x = x + expand_pos_embed
            x = rearrange(x, 'b (t n) c -> (b t) n c', t=self.time_steps)
        num_patches = self.encoder.patch_embed.num_patches
        if self.encoder_cls and self.decoder_cls:
            num_patches += 1
            x = torch.cat((
                x_cls_token,
                x
            ), dim=1)
        if self.camera_params_enabled:
            num_patches += 1
            cam_embed = self.camera_encoder(camera) #B, T, decoder_dim
            gt_embed = cam_embed[:, 1:, :]
            x = torch.cat((
                rearrange(cam_embed[:, :self.time_steps, :], 'b (t n) c -> (b t) n c', n=1),
                x
            ), dim=1)
        x = rearrange(x, '(b t) n c -> b (t n) c', t=self.time_steps)
        attn_mask = get_cvm_attn_mask(num_patches*(self.time_steps), self.time_steps).to(x.device)
        x = self.decoder(x, 0, attn_mask) #(B, N*(T-1), 768*16)

        if self.camera_params_enabled:
            pred_cams = x[1]
            x = x[0]

        x = rearrange(x, 'b (t n) c -> (b t) n c', t=self.time_steps)
        if self.encoder_cls and self.decoder_cls:
            if self.camera_params_enabled:
                x = rearrange(x[:, 2:, :], '(b t) n c -> b (t n) c', t=self.time_steps)
            else:
                x = rearrange(x[:, 1:, :], '(b t) n c -> b (t n) c', t=self.time_steps)
        elif self.camera_params_enabled:
            x = rearrange(x[:, 1:, :], '(b t) n c -> b (t n) c', t=self.time_steps)
        else:
            x = rearrange(x, '(b t) n c -> b (t n) c', t=self.time_steps)

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

def autoreg_vit_small_patch16_dec192d4(**kwargs):
    model = AutoregViTtimm(
        patch_size=16, encoder_embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 

def autoreg_beitv2_large_patch16_dec512d8(**kwargs):
    model = AutoregViTtimm(encoder_name='beitv2_large_patch16_224.in1k_ft_in22k_in1k', backbone_name='EncoderBeit', 
                        patch_size=16, encoder_embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                        decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

autoreg_vit_small_patch16 = autoreg_vit_small_patch16_dec192d4 # decoder: 192 dim, 4 blocks
autoreg_beit_large_patch16 = autoreg_beitv2_large_patch16_dec512d8
