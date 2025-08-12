import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from basic_cvar import AdaLNSABlock, SABlock, sample_with_top_k_top_p_


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C

class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class CVAR_Transformer(nn.Module):
    def __init__(
        self, vae_local,
        norm_eps=1e-6, aln=1, aln_gamma_init=1e-3, shared_aln=False, cond_drop_rate=0.1,
        depth=16, depth_cond=8, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        layer_scale=-1., tau=4, cos_attn=False,
        flash_if_available=True, fused_if_available=True,
        device='cpu'
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.dim_cbembed, vae_local.num_cbembed
        self.depth, self.depth_cond, self.C, self.D, self.num_heads = depth, depth_cond, embed_dim, embed_dim, num_heads
        self.using_aln, self.aln_init, self.aln_gamma_init, self.layer_scale = aln >= 0, aln, aln_gamma_init, layer_scale
        if self.using_aln and layer_scale != -1:
            print(f'**WARNING**: using AdaLNSABlock with {aln=:g}, {aln_gamma_init=:g}; the arg {layer_scale=:g} will be IGNORED because only SABlock cares about layer_scale', flush=True)
        self.device=device
        #self.prog_si = -1
        quant = vae_local.vq_layer
        self.vae_proxy = (vae_local,)
        self.vae_quant_proxy = (quant,)

        self.multi_points = quant.multi_points
        self.multi_points_nb = quant.multi_points_nb
        self.L = sum(pn for pn in self.multi_points_nb)
        self.first_l = self.multi_points_nb[0]

        
        self.num_stages_minus_1 = len(self.multi_points_nb) - 1
        self.rng = torch.Generator(device=device)
        
        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)
   
        # 2. sos embedding
        init_std = math.sqrt(1 / (self.C) / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        self.pos_embed = nn.Embedding(self.multi_points_nb[-1], self.C)
        nn.init.trunc_normal_(self.pos_embed.weight.data, mean=0, std=init_std)

        pos_idx = torch.cat([torch.tensor(A) for A in self.multi_points]).view(1, self.L)
        self.register_buffer('pos_idx', pos_idx)

        # 4. Scale (Level) embedding
        self.lvl_embed = nn.Embedding(len(self.multi_points_nb), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        d: torch.Tensor = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.multi_points_nb)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)

        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # Control Block
        attn_bias_for_masking_cond = torch.zeros((1, 1, self.L*2, self.L*2))
        d_bias_cond = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        attn_bias_for_masking_cond[:, :, self.L:self.L*2, self.L:self.L*2] = d_bias_cond
        self.register_buffer('attn_bias_for_masking_cond', attn_bias_for_masking_cond.contiguous())

        # 5. Pre(0)/Post(1) embedding
        #self.op_embed = nn.Embedding(1, self.C)
        #nn.init.trunc_normal_(self.op_embed.weight.data, mean=0, std=init_std)
        #pre_idx = torch.tensor([[0]*self.L])
        #self.register_buffer('pre_idx', pre_idx)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        #self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            SABlock(
                layer_scale=layer_scale,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        self.blocks_cond = nn.ModuleList([
            SABlock(
                layer_scale=layer_scale,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        self.cond_conv = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.C, self.C, bias=True) 
                for _ in range(len(self.multi_points_nb))
            ]) 
            for _ in range(depth_cond)
        ])

        for layer in self.cond_conv:
            for linear in layer:
                nn.init.constant_(linear.weight, 0)
                nn.init.constant_(linear.bias, 0)
        
        # 6. classifier head
        self.head_nm = MultiInpIdentity()
        self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.V))
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        h = h_or_h_and_residual
        return self.head(self.head_nm(h.float()).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer(
        self, B: int,
        g_seed: Optional[int] = None, top_k=0, top_p=0.0,
        more_smooth=False, pre_x_BLCv_wo_first_l=None
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]

        self.rng.manual_seed(g_seed)
        rng = self.rng

        # Post first
        PE_post = self.lvl_embed(self.lvl_1L) + self.pos_embed(self.pos_idx) #+ self.op_embed(self.post_idx)
        sos = self.pos_start.expand(B, self.first_l, -1)
        next_token_map = sos + PE_post[:, :self.first_l]

        # Pre (condition)
        pre_x_BLC = self.word_embed(pre_x_BLCv_wo_first_l.float())  #(B,1+2+..+282,C)
        pre_x_BLC += self.lvl_embed(self.lvl_1L) + self.pos_embed(self.pos_idx) #+ self.op_embed(self.pre_idx)

        #next_token_map = sos.clone()
        cur_L = 0

        f_hat = sos.new_zeros(B, self.multi_points_nb[-1], self.Cvae)
        
        #for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.multi_points_nb):   # si: i-th segment
            cur_L += pn

            x = next_token_map
            #x_BLC_cond = torch.cat([pre_x_BLC, x], dim=1)
            x_BLC_cond = pre_x_BLC.clone()
            for idx, (b,b_cond) in enumerate(zip(self.blocks, self.blocks_cond)):
                x = b(x=x, attn_bias=self.attn_bias_for_masking[:, :,:x.size(1),:x.size(1)])


                if idx < self.depth_cond:
                    #x_BLC_cond = b_cond(x=x_BLC_cond, attn_bias=self.attn_bias_for_masking[:,:,:x_BLC_cond.size(1),:x_BLC_cond.size(1)])
                    x_BLC_cond = b_cond(x=x_BLC_cond, attn_bias=None)
                    x_BLC_cond_controlnet = x_BLC_cond[:,:]

                    processed_slices = []
                    cur_L2=0
                    for si2, pn2 in enumerate(self.multi_points_nb[:si+1]):
                        cur_L2 += pn2
                        slice_part = x_BLC_cond_controlnet[:, cur_L2-pn2:cur_L2]
                        processed_slice = self.cond_conv[idx][si2](slice_part)
                        processed_slices.append(processed_slice)
                    x_BLC_cond_controlnet_processed = torch.cat(processed_slices, dim=1)

                    x = x+x_BLC_cond_controlnet_processed

            logits_BlV = self.get_logits(x)[:,cur_L-pn:cur_L]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            #idx_Bl = torch.argmax(logits_BlV, dim=-1).unsqueeze(-1)

            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            h_BChw = h_BChw.reshape(B, pn, self.Cvae)

            f_hat, next_h = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.multi_points_nb), f_hat, h_BChw)

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_h = self.word_embed(next_h) + PE_post[:, cur_L:cur_L + self.multi_points_nb[si+1]]

                next_token_map = torch.cat((next_token_map, next_h), dim=1)  # (B, L+1, C)
        
        #for b in self.blocks: b.attn.kv_caching(False)
        return f_hat
    
    def forward(self, pre_x_BLCv_wo_first_l, x_BLCv_wo_first_l) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed =  (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        # Post first
        sos = self.pos_start.expand(B, self.first_l, -1)
        x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1) #(B,1+2+..+282,C)
        x_BLC += self.lvl_embed(self.lvl_1L) + self.pos_embed(self.pos_idx) #+ self.op_embed(self.post_idx)

        # Pre (condition)
        pre_x_BLC = self.word_embed(pre_x_BLCv_wo_first_l.float())  #(B,1+2+..+282,C)
        pre_x_BLC += self.lvl_embed(self.lvl_1L) + self.pos_embed(self.pos_idx) #+ self.op_embed(self.pre_idx)

        # Attention masking
        attn_bias = self.attn_bias_for_masking[:, :, :, :]
        #attn_bias_cond = self.attn_bias_for_masking_cond[:, :, :, :]

        #x_BLC_cond = torch.cat([pre_x_BLC, x_BLC], dim=1)
        x_BLC_cond = pre_x_BLC.clone()
        
        for idx, (b,b_cond) in enumerate(zip(self.blocks, self.blocks_cond)):
            x_BLC = b(x=x_BLC, attn_bias=attn_bias)

            if idx < self.depth_cond:
                #x_BLC_cond = b_cond(x=x_BLC_cond, attn_bias=attn_bias)
                x_BLC_cond = b_cond(x=x_BLC_cond, attn_bias=None)
                x_BLC_cond_controlnet = x_BLC_cond[:,:]

                processed_slices = []
                cur_L=0
                for si, pn in enumerate(self.multi_points_nb):
                    cur_L += pn
                    #x_BLC_cond_controlnet[:,cur_L-pn:cur_L]=self.cond_conv[idx][si](x_BLC_cond_controlnet[:,cur_L-pn:cur_L])
                    slice_part = x_BLC_cond_controlnet[:, cur_L-pn:cur_L]
                    processed_slice = self.cond_conv[idx][si](slice_part)
                    processed_slices.append(processed_slice)
                x_BLC_cond_controlnet_processed = torch.cat(processed_slices, dim=1)

                x_BLC = x_BLC+x_BLC_cond_controlnet_processed
            
        x_BLC = self.get_logits(x_BLC.float())
        
        return x_BLC    
    
    def special_init(self, hd0: float): # hd0: head init scale
        if hd0 >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(hd0)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(hd0)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            if True:
                self.head_nm.ada_lin[-1].weight.data.mul_(self.aln_init)
                if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                    self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: Union[AdaLNSABlock, SABlock]
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(self.aln_gamma_init)
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(self.aln_init)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2].mul_(self.aln_gamma_init)
                sab.ada_gss.data[:, :, 2:].mul_(self.aln_init)
    
    def extra_repr(self):
        gamma2_last = self.gamma2_last
        if isinstance(gamma2_last, nn.Parameter):
            gamma2_last = f'<vector {self.layer_scale}>'
        return f'drop_path_rate={self.drop_path_rate:g}, layer_scale={self.layer_scale:g}, gamma2_last={gamma2_last}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

def CVAR_XL(**kwargs):
    return CVAR_Transformer(depth=28, embed_dim=1152, num_heads=16, **kwargs)

def CVAR_L(**kwargs):
    return CVAR_Transformer(depth=24, embed_dim=1024, num_heads=16, **kwargs)

def CVAR_B(**kwargs):
    return CVAR_Transformer(depth=12, embed_dim=768, num_heads=12, **kwargs)

def CVAR_S(**kwargs):
    return CVAR_Transformer(depth=12, embed_dim=384, num_heads=6, **kwargs)

def CVAR_T(**kwargs):
    return CVAR_Transformer(depth=12, embed_dim=192, num_heads=3, **kwargs)

CVAR_Transformer_models = {
    'CVAR-XL': CVAR_XL,  
    'CVAR-L':  CVAR_L,
    'CVAR-B':  CVAR_B,
    'CVAR-S':  CVAR_S,
    'CVAR-T':  CVAR_T
}