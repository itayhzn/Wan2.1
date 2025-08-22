
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as TF
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention
from .model import rope_params, sinusoidal_embedding_1d, MLPProj, Head, WanAttentionBlock, WanLayerNorm, WanSelfAttention, WanT2VCrossAttention, WanRMSNorm, rope_apply

from datetime import datetime

import os

from utils import save_tensors

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


class PairedWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x1, x2, grid_sizes, edit_context, subject_context, seq_lens, freqs, original_x1=None, original_x2=None, should_edit=False, subject_masks=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x1.shape[:2], self.num_heads, self.head_dim

        q1, k1, v1 = self.qkv_fn(x1) # [B, F*H*W, n, d]
        
        x1 = flash_attention(
            q=rope_apply(q1, grid_sizes, freqs),
            k=rope_apply(k1, grid_sizes, freqs),
            v=v1,
            k_lens=seq_lens,
            window_size=self.window_size) # [B, F*H*W, n, d]

        if should_edit:
            # subject_masks = subject_masks.view(1, -1, 1, 1)
            # q2, k2, v2 = self.qkv_fn(x2) # [B, F*H*W, n, d]
            # q_edit, k_edit, v_edit = self.qkv_fn(edit_context) # [B, L2, n, d]

            # x2 = flash_attention(
            #     q=rope_apply(q1 * (1 - subject_masks) + q2 * subject_masks, grid_sizes, freqs),
            #     k=rope_apply(k1 * (1 - subject_masks) + k2 * subject_masks, grid_sizes, freqs),
            #     v=v1,
            #     k_lens=seq_lens,
            #     window_size=self.window_size) # [B, F*H*W, n, d]
            x2 = x1
        else:
            x2 = x1



        # output
        x1 = x1.flatten(2)
        x1 = self.o(x1)

        x2 = x2.flatten(2)
        x2 = self.o(x2)
        return x1, x2

    def qkv_fn(self, x):
        b, n, d = *x.shape[:1], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(x)).view(b, -1, n, d)
        v = self.v(x).view(b, -1, n, d)
        return q, k, v

class PairedWanT2VCrossAttention(PairedWanSelfAttention):

    def forward(self, x1, x2, grid_sizes, context1, context2, edit_context, subject_context, context_lens, save_tensors_dir=None, should_edit=False, original_x1=None, original_x2=None, subject_masks=None):
        r"""
        Args:
            x1, x2(Tensor): Shape [B, L1, C]
            context1, context2(Tensor): Shape [B, L2, C]
            edit_context(Tensor): Shape [B, L3, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x1.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q1 = self.norm_q(self.q(x1)).view(b, -1, n, d)
        k_context1 = self.norm_k(self.k(context1)).view(b, -1, n, d)
        v_context1 = self.v(context1).view(b, -1, n, d)

        q2 = self.norm_q(self.q(x2)).view(b, -1, n, d)
        k_edit = self.norm_k(self.k(edit_context)).view(b, -1, n, d)
        v_edit = self.v(edit_context).view(b, -1, n, d)


        # compute attention
        x1 = flash_attention(q1, k_context1, v_context1) # , k_lens=context_lens

        x2_context = flash_attention(q2, k_context1, v_context1) # , k_lens=context_lens

        if should_edit:
            x2_edit = flash_attention(q2, k_edit, v_edit) # , k_lens=context_lens
            print(f"x2_edit: {x2_edit.shape}") # DEBUG --- IGNORE ---
            print(f"subject_masks: {subject_masks.shape}") # DEBUG --- IGNORE ---
            x2 = x2_context * (1 - subject_masks) + x2_edit * subject_masks
        else:
            x2 = x2_context

        # output
        x1 = x1.flatten(2)
        x1 = self.o(x1)

        x2 = x2.flatten(2)
        x2 = self.o(x2)

        return x1, x2
    
    

class PairedWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = PairedWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = PairedWanT2VCrossAttention(dim,
                                                    num_heads,
                                                    (-1, -1),
                                                    qk_norm,
                                                    eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x1,
        x2,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context1,
        context2,
        edit_context,
        subject_context,
        context_lens,
        save_tensors_dir=None,
        should_edit=False,
        original_x1=None,
        original_x2=None,
        subject_masks=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y1, y2 = self.self_attn(
            self.norm1(x1).float() * (1 + e[1]) + e[0],
            self.norm1(x2).float() * (1 + e[1]) + e[0], 
            grid_sizes,
            edit_context,
            subject_context,
            seq_lens,
            freqs, 
            original_x1=original_x1,
            original_x2=original_x2,
            should_edit=should_edit,
            subject_masks=subject_masks)
        with amp.autocast(dtype=torch.float32):
            x1 = x1 + y1 * e[2]
            x2 = x2 + y2 * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x1, x2, grid_sizes, context1, context2, edit_context, subject_context, context_lens, e, save_tensors_dir, should_edit, original_x1, original_x2, subject_masks):
            _x1, _x2 = self.cross_attn(self.norm3(x1), self.norm3(x2), grid_sizes, context1, context2, edit_context, subject_context, context_lens, save_tensors_dir, should_edit, original_x1, original_x2, subject_masks=subject_masks)
            x1 = x1 + _x1
            x2 = x2 + _x2
            y1 = self.ffn(self.norm2(x1).float() * (1 + e[4]) + e[3])
            y2 = self.ffn(self.norm2(x2).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x1 = x1 + y1 * e[5]
                x2 = x2 + y2 * e[5]
            return x1, x2

        x1, x2 = cross_attn_ffn(x1, x2, grid_sizes, context1, context2, edit_context, subject_context, context_lens, e, save_tensors_dir, should_edit, original_x1, original_x2, subject_masks=subject_masks)
        return x1, x2
    
    def qkv_fn(self, x):
        return self.self_attn.qkv_fn(x)

class PairedWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'

        # swap one attention block at position my_attn_pos to MyAttentionBlock 
        self.blocks = nn.ModuleList([
            PairedWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()


    def forward(
        self,
        x1, x2,
        t,
        context1, context2,
        seq_len,
        clip_fea=None,
        y1=None, y2=None,
        edit_context=None,
        subject_context=None,
        save_tensors_dir=None,
        should_edit=False,
        subject_masks=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x1, x2 (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context1, context2 (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y1, y2 (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            edit_context (Tensor, *optional*):
                List of additional context embeddings, shape [L_edit, C]

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y1 is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        original_x1 = x1
        original_x2 = x2

        if y1 is not None:
            x1 = [torch.cat([u, v], dim=0) for u, v in zip(x1, y1)]
        if y2 is not None:
            x2 = [torch.cat([u, v], dim=0) for u, v in zip(x2, y2)]  

        # embeddings
        x1 = [self.patch_embedding(u.unsqueeze(0)) for u in x1]
        x2 = [self.patch_embedding(u.unsqueeze(0)) for u in x2]
        
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x1])
        
        x1 = [u.flatten(2).transpose(1, 2) for u in x1]
        x2 = [u.flatten(2).transpose(1, 2) for u in x2]

        seq_lens = torch.tensor([u.size(1) for u in x1], dtype=torch.long)
    
        assert seq_lens.max() <= seq_len 
        
        x1 = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x1
        ])
        x2 = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x2
        ])
        
        ########################################
        if subject_masks is not None:
            subject_masks = torch.stack([ 
                TF.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), # [1, 1, F, H, W]
                size=(grid_sizes[i][0].item(), grid_sizes[i][1].item(), grid_sizes[i][2].item()),
                mode='trilinear',
                align_corners=False
                ).view(1, -1, 1) # [1, 1, F', H', W'] -> [1, F'*H'*W', 1]
            for i, mask in enumerate(subject_masks)]).to(x1.device) # [B, F'*H'*W', 1]

        ########################################


        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context1 = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context1
            ]))
        context2 = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context2
            ]))
        if edit_context is not None:
            edit_context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in edit_context
                ]))
            
        if subject_context is not None:
            subject_context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in subject_context
                ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context1 = torch.concat([context_clip, context1], dim=1)
            context2 = torch.concat([context_clip, context2], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context1=context1,
            context2=context2,
            edit_context=edit_context,
            subject_context=subject_context,
            context_lens=context_lens,
            save_tensors_dir=None,
            should_edit=should_edit,
            original_x1=original_x1,
            original_x2=original_x2,
            subject_masks=subject_masks
            )

        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                kwargs['save_tensors_dir'] = save_tensors_dir

            x1, x2 = block(x1, x2, **kwargs)

            if i == len(self.blocks) - 1:
                kwargs['save_tensors_dir'] = None

        # head
        x1 = self.head(x1, e) 
        x2 = self.head(x2, e)

        # unpatchify
        x1 = self.unpatchify(x1, grid_sizes) 
        x2 = self.unpatchify(x2, grid_sizes)
        return [u.float() for u in x1], [u.float() for u in x2]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def prepare_for_qkv(self, x):
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    
        assert seq_lens.max() <= seq_lens
        
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        return x, grid_sizes

    def qkv_fn(self, x):
        return self.blocks[0].qkv_fn(x)

    def compute_grid_sizes(self, x):
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        return torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])