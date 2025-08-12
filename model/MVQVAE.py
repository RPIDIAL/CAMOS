# Delete Decoder_pos_embed
# --------------------------------------------------------

import torch
import torch.nn as nn

import numpy as np

from timm.models.vision_transformer import Attention, Mlp
from torch.nn import functional as F

from batchgenerators.utilities.file_and_folder_operations import *

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PointEmbedder(nn.Module):
    def __init__(self, hidden_size, out_size, bias=True):
        super().__init__()
        self.proj = nn.Linear(hidden_size, out_size, bias=bias)

    def forward(self, p):
        p = self.proj(p)
        return p

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        x = self.tanh(x)
        return x


class MVQVAE_Transformer(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        hidden_size=1152,
        num_cbembed=512,
        dim_cbembed=64,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        beta=0.25,
        dist_typ='euc',
        scale=10,
        device='cpu'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_cbembed = num_cbembed
        self.dim_cbembed = dim_cbembed

        #####//#####
        self.p_embedder = PointEmbedder(in_channels, hidden_size, bias=True)
        self.pos_embed = nn.Parameter(torch.randn(1, input_size, hidden_size))

        self.encoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.encod_emb = nn.Linear(hidden_size, dim_cbembed)
        self.decod_emb = nn.Linear(dim_cbembed, hidden_size)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

        self.vq_layer = MVectorQuantizer(num_cbembed, dim_cbembed, beta, self.pos_embed, dist_typ, scale, device)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def img_to_idxBl(self, x):
        x = self.p_embedder(x) + self.pos_embed
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.encod_emb(x)
        x = self.vq_layer.f_to_idxBl_or_fhat(x, to_fhat=False)
        return x

    def idxBl_to_h(self, gt_ms_idx_Bl): 
        return self.vq_layer.idxBl_to_var_input(gt_ms_idx_Bl)

    def idxBl_to_h_cond(self, gt_ms_idx_Bl): 
        return self.vq_layer.idxBl_to_var_input_cond(gt_ms_idx_Bl)

    def img_to_recon(self, x, nb=-1):
        x = self.p_embedder(x) + self.pos_embed
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.encod_emb(x)
        quantized_inputs = self.vq_layer.f_to_idxBl_or_fhat(x, to_fhat=True)
        quantized_inputs = self.decod_emb(quantized_inputs[nb])

        z = quantized_inputs  + self.pos_embed
        for decoder_block in self.decoder_blocks:
            z = decoder_block(z)
        recon = self.final_layer(z)

        return recon

    def forward(self, x):
        # Encoder
        x = self.p_embedder(x) + self.pos_embed
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.encod_emb(x)

        # VQ layer
        quantized_inputs, vq_loss = self.vq_layer(x)
        quantized_inputs = self.decod_emb(quantized_inputs)
        
        # Decoder
        z = quantized_inputs  + self.pos_embed
        for decoder_block in self.decoder_blocks:
            z = decoder_block(z)
        recon = self.final_layer(z)

        return recon, vq_loss, x

#################################################################################
#                                   MVQVAET Configs                                  #
#################################################################################

def MVQVAET_XL(**kwargs):
    return MVQVAE_Transformer(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def MVQVAET_L(**kwargs):
    return MVQVAE_Transformer(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def MVQVAET_B(**kwargs):
    return MVQVAE_Transformer(depth=12, hidden_size=768, num_heads=12, **kwargs)

def MVQVAET_S(**kwargs):
    return MVQVAE_Transformer(depth=12, hidden_size=384, num_heads=6, **kwargs)

def MVQVAET_T(**kwargs):
    return MVQVAE_Transformer(depth=12, hidden_size=192, num_heads=3, **kwargs)

MVQVAE_Transformer_models = {
    'MVQVAET-XL': MVQVAET_XL,  
    'MVQVAET-L':  MVQVAET_L,
    'MVQVAET-B':  MVQVAET_B,
    'MVQVAET-S':  MVQVAET_S,
    'MVQVAET-T':  MVQVAET_T
}

class MVectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25, pos_embed=None, dist_typ='euc', scale=10, device='cpu'):
        super(MVectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        ######## NEW ########
        #optimal/src/MVQVAE_preprocessing.ipynb
        self.multi_points=[]
        data=load_json('template_ms8_idx_v7.json')

        self.multi_points_nb=data['multi_points']
        for mnb in self.multi_points_nb:
            self.multi_points.append(data[str(mnb)])

        self.SN = len(self.multi_points)
        self.template_pos=torch.tensor(np.load('template_pos.npy')).to(device)
        self.euc_dist=torch.tensor(np.load('template_euc.npy')).to(device)
        self.geo_dist=torch.tensor(np.load('template_geo.npy')).to(device)
        ######## NEW ########
        self.pos_embed = pos_embed
        self.dist_typ=dist_typ
        self.device=device

    def point_interpolate(self,
        feat: torch.Tensor,
        bef_idx: torch.Tensor,
        aft_idx: torch.Tensor
    ):
        
        if self.dist_typ=='euc':
            dist=self.euc_dist
        elif self.dist_typ=='geo':
            dist=self.geo_dist
        elif self.dist_typ=='cos':
            pos_embed_detached = self.pos_embed.detach().squeeze(0)
            pos_embed_normed = pos_embed_detached / pos_embed_detached.norm(dim=1, keepdim=True)
            similarity_matrix = torch.mm(pos_embed_normed, pos_embed_normed.t())
            dist=1-similarity_matrix
        else:
            print('distance_type error')

        dist = torch.clamp(dist, min=1e-8)
        selected_dist=dist[bef_idx[:,None], aft_idx]
        weights = 1.0 / selected_dist
        weights /= weights.sum(dim=0, keepdim=True)

        #torch.matmul(feat,weights).T
        tmp = torch.matmul(feat.transpose(-2, -1), weights.float())
        outs_bmm = tmp.transpose(-2, -1)

        return outs_bmm

    def f_to_idxBl_or_fhat(self, f_BNC, to_fhat: bool):
        B, N, C  = f_BNC.shape
        f_no_grad = f_BNC.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl=[]

        for idx, pnb in enumerate(self.multi_points):
            z_NC = self.point_interpolate(f_rest, torch.tensor(self.multi_points[-1]), torch.tensor(pnb)).reshape(-1, C).float() if (idx != self.SN-1) else f_rest.reshape(-1, C)
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)

            idx_BN = idx_N.view(B, -1)
            h_BNC = self.point_interpolate(self.embedding(idx_BN), torch.tensor(pnb), torch.tensor(self.multi_points[-1])).reshape(B,-1, C).float() if (idx != self.SN-1) else self.embedding(idx_BN).reshape(B,-1, C)

            f_hat = f_hat + h_BNC
            f_rest -= h_BNC

            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, -1))
        return f_hat_or_idx_Bl

    def idxBl_to_var_input(self, gt_ms_idx_Bl):
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.D
        SN = len(self.multi_points)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, 282, C, dtype=torch.float32)
        pn_next = self.multi_points[0]

        for si in range(self.SN-1):

            h_BChw = self.point_interpolate(self.embedding(gt_ms_idx_Bl[si]), torch.tensor(self.multi_points[si]), torch.tensor(self.multi_points[-1])).reshape(B,-1, C).float()
            f_hat.add_(h_BChw)
            pn_next = self.multi_points[si+1]
            next_scales.append(self.point_interpolate(f_hat, torch.tensor(self.multi_points[-1]), torch.tensor(pn_next)).reshape(B,-1, C).float())
        return next_scales

    def idxBl_to_var_input_cond(self, gt_ms_idx_Bl):
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.D
        SN = len(self.multi_points)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, 282, C, dtype=torch.float32)
        pn_next = self.multi_points[0]

        for si in range(self.SN):
            h_BChw = self.point_interpolate(self.embedding(gt_ms_idx_Bl[si]), torch.tensor(self.multi_points[si]), torch.tensor(self.multi_points[-1])).reshape(B,-1, C).float()
            f_hat.add_(h_BChw)
            pn_next = self.multi_points[si]
            next_scales.append(self.point_interpolate(f_hat, torch.tensor(self.multi_points[-1]), torch.tensor(pn_next)).reshape(B,-1, C).float())
        return next_scales

    def forward(self, f_BNC):

        B, N, C  = f_BNC.shape
        f_no_grad = f_BNC.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        mean_vq_loss: torch.Tensor = 0.0

        for idx, pnb in enumerate(self.multi_points):
            rest_NC = self.point_interpolate(f_rest, torch.tensor(self.multi_points[-1]), torch.tensor(pnb)).reshape(-1, C).float() if (idx != self.SN-1) else f_rest.reshape(-1, C)
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_BN = idx_N.view(B, -1)
            h_BNC = self.point_interpolate(self.embedding(idx_BN), torch.tensor(pnb), torch.tensor(self.multi_points[-1])).reshape(B,-1, C).float() if (idx != self.SN-1) else self.embedding(idx_BN).reshape(B,-1, C)

            f_hat = f_hat + h_BNC
            f_rest -= h_BNC

            mean_vq_loss += F.mse_loss(f_hat.data, f_BNC).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
        mean_vq_loss *= 1. / self.SN
        f_hat = (f_hat.data - f_no_grad).add_(f_BNC)

        return f_hat, mean_vq_loss  # [B x N x D]

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor): # only used in VAR inference

        B,N,_=f_hat.shape
        if si != SN-1:
            h = self.point_interpolate(h_BChw, torch.tensor(self.multi_points[si]), torch.tensor(self.multi_points[-1])).reshape(B,-1, self.D).float()
            f_hat.add_(h)
            next_h = self.point_interpolate(f_hat, torch.tensor(self.multi_points[-1]), torch.tensor(self.multi_points[si+1])).reshape(B, -1, self.D).float()

            return f_hat, next_h
        else:
            f_hat.add_(h_BChw)
            return f_hat, f_hat