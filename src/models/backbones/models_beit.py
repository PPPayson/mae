import math
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, SwiGLU, LayerNorm, DropPath, trunc_normal_, use_fused_attn
from timm.layers import resample_patch_embed, resample_abs_pos_embed, resize_rel_pos_bias_table, ndgrid
from einops import rearrange

def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(ndgrid(torch.arange(window_size[0]), torch.arange(window_size[1])))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index

def gen_relative_temporal_index(num_frames: int) -> torch.Tensor:
    num_relative_distance = 2 * num_frames - 1
    coords = torch.arange(num_frames)
    relative_coords = coords[:, None] - coords[None, :] # t, t
    relative_coords += num_frames - 1 # shift to start from 0
    relative_temporal_index = torch.zeros(size=(num_frames, num_frames), dtype=relative_coords.dtype)
    relative_temporal_index[:, :] = relative_coords # No CLS token
    return relative_temporal_index

class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            window_size: Optional[Tuple[int, int]] = None,
            attn_head_dim: Optional[int] = None,
            temporal_embed: Optional[bool] = False,
            num_frames: Optional[int] = 1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            self.register_buffer("relative_position_index", gen_relative_position_index(window_size), persistent=False)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        
        self.temporal_embed = temporal_embed
        if temporal_embed:
            self.num_frames=num_frames
            self.num_rel_temp_distance = 2 * num_frames - 1
            self.relative_temporal_bias_table = nn.Parameter(
                torch.zeros(self.num_rel_temp_distance, num_heads)) # 2*t-1, nH
            self.register_buffer("relative_temporal_index", gen_relative_temporal_index(num_frames), persistent=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def _get_rel_temp_bias(self):
        relative_temporal_bias = self.relative_temporal_bias_table[self.relative_temporal_index.view(-1)].view(self.num_frames, self.num_frames, -1)
        relative_temporal_bias = relative_temporal_bias.permute(2, 0, 1).contiguous() #nH, t, t
        return relative_temporal_bias.unsqueeze(0)

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None, attn_mask=None, num_frames=1, f2d=False):
        B, N, C = x.shape

        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            rel_embed = self._get_rel_pos_bias()
            if not f2d:
                rel_embed = rel_embed.repeat(1, 1, num_frames, num_frames)
            if attn.shape[2] != rel_embed.shape[2]:
                rel_embed = rel_embed[:, :, :attn.shape[2], :attn.shape[3]]
            attn = attn + rel_embed            
            # attn = attn + self._get_rel_pos_bias()
        if self.temporal_embed:
            temp_embed = self._get_rel_temp_bias()
            if f2d:
                temp_embed = temp_embed[:, :, :1, :1]
                temp_embed = torch.repeat_interleave(temp_embed, repeats=attn.shape[2], dim=2, output_size=attn.shape[2])
                temp_embed = torch.repeat_interleave(temp_embed, repeats=attn.shape[3], dim=3, output_size=attn.shape[3])
            else:
                temp_embed = torch.repeat_interleave(temp_embed, repeats=int(attn.shape[2]/num_frames), dim=2, output_size=attn.shape[2])
                temp_embed = torch.repeat_interleave(temp_embed, repeats=int(attn.shape[3]/num_frames), dim=3, output_size=attn.shape[3])
            attn = attn + temp_embed
        if shared_rel_pos_bias is not None:
            attn = attn + shared_rel_pos_bias
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            mlp_ratio: float = 4.,
            scale_mlp: bool = False,
            swiglu_mlp: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            window_size: Optional[Tuple[int, int]] = None,
            attn_head_dim: Optional[int] = None,
            temporal_embed: Optional[bool] = False,
            num_frames: Optional[int] = 1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
            temporal_embed=temporal_embed,
            num_frames=num_frames,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        if swiglu_mlp:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None, attn_mask=None, num_frames=1, f2d=False):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias, attn_mask=attn_mask, num_frames=num_frames, f2d=f2d))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias, attn_mask=attn_mask, num_frames=num_frames, f2d=f2d))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size))

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1, self.window_area + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        
class Beit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            use_abs_pos_emb: bool = True,
            use_rel_pos_bias: bool = False,
            use_shared_rel_pos_bias: bool = False,
            head_init_scale: float = 0.001,
            pos_embed: str = '2d',
            time_steps: int = 1,
            use_masked_im_modeling: bool = False,
            **kwargs
    ):
        super().__init__()
        # self.time_steps = time_steps
        # self.pos_embed_type = pos_embed
        self.patch_size = patch_size
        self.use_masked_im_modeling = use_masked_im_modeling
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False
        if use_masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))
        else:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=False)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=False,
            strict_img_size=False
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.grid_size,
                num_heads=num_heads,
            )
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                scale_mlp=scale_mlp,
                swiglu_mlp=swiglu_mlp,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,
                temporal_embed=(pos_embed=='rel_3d'),
                num_frames=time_steps,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
            elif 'relative_temporal_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed|rel_pos_bias',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if an int, if is a sequence, select by matching indices
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    # def forward_features(self, x):
    #     x = self.patch_embed(x)
    #     x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     if self.pos_embed is not None:
    #         x = x + self.pos_embed
    #     x = self.pos_drop(x)

    #     rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    #     for blk in self.blocks:
    #         if self.grad_checkpointing and not torch.jit.is_scripting():
    #             x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
    #         else:
    #             x = blk(x, shared_rel_pos_bias=rel_pos_bias)
    #     x = self.norm(x)
    #     return x

    # def forward_head(self, x, pre_logits: bool = False):
    #     if self.global_pool:
    #         x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    #     x = self.fc_norm(x)
    #     x = self.head_drop(x)
    #     return x if pre_logits else self.head(x)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head_drop(self.fc_norm(x))
        
    def forward_features(self, x, attn_mask=None, num_frames=1, pool=True, f2d=False, temporal_embed=None, return_all_tokens=False, mask=None):
        # if f2d and temporal_embed is not None:
        #     temporal_embed = torch.zeros(temporal_embed[:, :1, :, :].shape).to(x.device)

        if self.use_masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if not f2d:
            x = rearrange(x, '(b t) n c -> b (t n) c', t=num_frames)

        for blk in self.blocks:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias, attn_mask=attn_mask, num_frames=num_frames, f2d=f2d)
        x = self.norm(x)
        if pool and not return_all_tokens:
            if not f2d:

                x = rearrange(x, 'b (t n) c -> (b t) n c', t=num_frames)
                x = self.fc_norm(x[:, self.num_prefix_tokens:].mean(dim=1)) if self.global_pool == 'avg' else x[:, 0]
                return x
            x = self.fc_norm(x[:, self.num_prefix_tokens:].mean(dim=1)) if self.global_pool == 'avg' else x[:, 0]
            return x
        else:
            if not f2d:
                x = rearrange(x, 'b (t n) c -> (b t) n c', t=num_frames)
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))
            return x
        

    def forward(self, x, attn_mask=None, num_frames=1, pool=True, f2d=False, temporal_embed=None, mask=None):
        return self.forward_features(x, attn_mask, num_frames, pool, f2d, temporal_embed, mask=mask)


    def prepare_tokens(self, x, mask=None):
        x = self.patch_embed(x)

        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        return x

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x    