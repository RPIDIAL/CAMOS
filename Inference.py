
import torch

import logging
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *

from dataset.face_dataset import FaceDataset
from model.MVQVAE import MVQVAE_Transformer_models
from model.CVAR import CVAR_Transformer_models
#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if 1:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device='cuda:{}'.format(args.gpu)
    torch.manual_seed(args.random_seed)
    torch.cuda.set_device(device)

    # Create model:
    MVQVAE = MVQVAE_Transformer_models[args.mvae_model](
        input_size=args.point_size,
        in_channels=3,
        num_cbembed=args.mvae_num_cbembed,
        dim_cbembed=args.mvae_dim_cbembed,
        dist_typ=args.mvae_dist_typ,
        scale=args.mvae_scale,
        device=device
    ).to(device)

    #print('MVQVAE Load: ',mvae_ckpt_path)
    mvae_ckpt_path=args.mvqvae_path
    mvae_state_dict = torch.load(mvae_ckpt_path, map_location=lambda storage, loc: storage)
    new_mvae_state_dict_model = {}
    for k, v in mvae_state_dict['model'].items():
        name = k[7:] if k.startswith('module.') else k 
        new_mvae_state_dict_model[name] = v
    MVQVAE.load_state_dict(new_mvae_state_dict_model)
    requires_grad(MVQVAE, False)
    print('MVQVAE Load: ',mvae_ckpt_path)
    MVQVAE.eval()

    CVAR = CVAR_Transformer_models[args.cvar_model](
        vae_local=MVQVAE,
        depth_cond=args.cvar_depth_cond,
        device=device
    ).to(device)

    cvar_ckpt_path=args.cvar_path
    cvar_state_dict = torch.load(cvar_ckpt_path, map_location=lambda storage, loc: storage)
    new_cvar_state_dict_model = {}
    for k, v in cvar_state_dict['cvar'].items():
        name = k[7:] if k.startswith('module.') else k 
        new_cvar_state_dict_model[name] = v
    CVAR.load_state_dict(new_cvar_state_dict_model)
    print('cvar Load: ',cvar_ckpt_path)
    CVAR.eval()

    data_json1=load_json(args.data_path1)
    data_json2=load_json(args.data_path2)
    
    test_list=data_json1[str(args.fold)]['test']+data_json2[str(args.fold)]['test']

    test_dataset=FaceDataset(test_list, sym=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_list))

    dat={}
    with torch.no_grad():
        torch.manual_seed(args.random_seed)

        for data in test_loader:
            with torch.no_grad():
                complete=data['complete'].to(device).float()                      
                labels_list = MVQVAE.img_to_idxBl(complete)                       
                input_h_list = MVQVAE.idxBl_to_h(labels_list)                               

                cond=data['cond'].to(device).float()                              
                pre_labels_list = MVQVAE.img_to_idxBl(cond)                       
                pre_input_h_list = MVQVAE.idxBl_to_h_cond(pre_labels_list)        
                pre_x_BLCv_wo_first_l = torch.concat(pre_input_h_list, dim=1)     

                quantized_inputs=CVAR.autoregressive_infer(cond.size(0), g_seed = args.random_seed, top_k=1, top_p=0, pre_x_BLCv_wo_first_l=pre_x_BLCv_wo_first_l)
                quantized_inputs = MVQVAE.decod_emb(quantized_inputs)
                z = quantized_inputs + MVQVAE.pos_embed
                for decoder_block in MVQVAE.decoder_blocks:
                    z = decoder_block(z)
                recon = MVQVAE.final_layer(z)

            for i in range(cond.shape[0]):
                g={}
                g['0']=recon[i,:,:3].tolist()
                g['x_mean'] = data['x_mean'][i].tolist()
                g['y_mean'] = data['y_mean'][i].tolist()
                g['z_mean'] = data['z_mean'][i].tolist()
                g['all_std'] = data['all_std'][i].tolist()
                g['complete'] = data['complete'][i].tolist()
                g['cond'] = data['cond'][i].tolist()

                pid = data['pid'][i].split('/')[-1]
                dat[pid] = g
                save_p=f'{args.save_path}/{pid}.json'
                print(save_p)

                save_json(dat[pid], save_p)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mvqvae_path", type=str)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--data_path1", type=str)
    parser.add_argument("--data_path2", type=str)
    parser.add_argument("--save_path", type=str)

    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--cvar_model", type=str, default='CVAR-T')
    parser.add_argument("--cvar_depth_cond", type=int, default=10)

    parser.add_argument("--point_size", type=int, default=282)
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--mvae_results_dir", type=str)
    parser.add_argument("--mvae_model", type=str, default='VQVAET-T')
    parser.add_argument("--mvae_num_cbembed", type=int, default=512)
    parser.add_argument("--mvae_dim_cbembed", type=int, default=64)
    parser.add_argument("--mvae_dist_typ", type=str, default='geo')
    parser.add_argument("--mvae_scale", type=int, default=8)

    parser.add_argument('-g', '--gpu', type=int, default=0)

    args = parser.parse_args()
    for f in range(5):
        args.fold=f
        main(args)