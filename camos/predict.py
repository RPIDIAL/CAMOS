"""Model loading and landmark prediction (multi-scale VQ-VAE tokens + conditional VAR)."""
import os

import numpy as np
import torch

# Greedy autoregressive decoding is sensitive to tiny numerical differences (near-tie tokens),
# so reduced-precision matmuls (TF32) are disabled to keep CPU and GPU results identical.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from model import MVQVAE_Transformer_models, CVAR_Transformer_models

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def _strip(state):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def load_models(mvqvae_path, cvar_path, device="cpu", mvae_model="MVQVAET-T", cvar_model="CVAR-T",
                num_cbembed=256, dim_cbembed=64, scale=8, dist_typ="geo", depth_cond=10, point_size=282):
    """Build the networks with the published CAMOS configuration and load the checkpoints."""
    mvqvae = MVQVAE_Transformer_models[mvae_model](
        input_size=point_size, in_channels=3, num_cbembed=num_cbembed, dim_cbembed=dim_cbembed,
        dist_typ=dist_typ, scale=scale, device=device, template_dir=ASSET_DIR).to(device)
    ck = torch.load(mvqvae_path, map_location="cpu")
    mvqvae.load_state_dict(_strip(ck["model"]))
    mvqvae.eval()
    for p in mvqvae.parameters():
        p.requires_grad = False

    cvar = CVAR_Transformer_models[cvar_model](vae_local=mvqvae, depth_cond=depth_cond, device=device).to(device)
    ck = torch.load(cvar_path, map_location="cpu")
    cvar.load_state_dict(_strip(ck["cvar"]))
    cvar.eval()
    return mvqvae, cvar


@torch.no_grad()
def predict_normalized(mvqvae, cvar, cond_norm282, device="cpu", seed=0):
    """cond_norm282: (282, 3) normalized pre-operative landmarks -> (282, 3) normalized prediction."""
    torch.manual_seed(seed)
    cond = torch.tensor(np.asarray(cond_norm282), dtype=torch.float32, device=device)[None]
    pre_labels = mvqvae.img_to_idxBl(cond)
    pre_h = torch.cat(mvqvae.idxBl_to_h_cond(pre_labels), dim=1)
    f_hat = cvar.autoregressive_infer(1, g_seed=seed, top_k=1, top_p=0, pre_x_BLCv_wo_first_l=pre_h)
    z = mvqvae.decod_emb(f_hat) + mvqvae.pos_embed
    for block in mvqvae.decoder_blocks:
        z = block(z)
    recon = mvqvae.final_layer(z)
    return recon[0, :, :3].cpu().numpy().astype(float)
