import os
import pytz
import torch
import logging
import argparse

from time import time
from datetime import datetime

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.nn import functional as F
from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

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

def est_time_formatter(record, datefmt="%Y-%m-%d %H:%M:%S"):
    est = pytz.timezone('US/Eastern')
    return datetime.fromtimestamp(record.created, est).strftime(datefmt)

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

def extract_submodule_state_dict(full_state_dict, submodule_prefix):
    submodule_state_dict = {k[len(submodule_prefix)+1:]: v for k, v in full_state_dict.items() if k.startswith(submodule_prefix)}
    return submodule_state_dict
def model_load(module, module_state, trainable=False):
    if isinstance(module, torch.nn.Parameter) or isinstance(module, torch.Tensor):
        module.data = module_state.clone()
        if not trainable:
            module.requires_grad = False
    else:
        module.load_state_dict(module_state, strict=False)
        if trainable==False:
            for param in module.parameters():
                param.requires_grad = False
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device='cuda:{}'.format(args.gpu)
    torch.manual_seed(args.random_seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    experiment_dir=args.experiment_dir
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

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

    pt_ckpt_path=args.pretrained_path
    pt_state_dict = torch.load(pt_ckpt_path, map_location=lambda storage, loc: storage)

    new_state_dict_model = {}
    for k, v in pt_state_dict['cvar'].items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict_model[name] = v

    CVAR.load_state_dict(new_state_dict_model)
    for param in CVAR.parameters():
        param.requires_grad = False
    for param in CVAR.blocks_cond.parameters():
        param.requires_grad = True
    for param in CVAR.cond_conv.parameters():
        param.requires_grad = True

    opt = torch.optim.AdamW(CVAR.parameters(), lr=args.lr)
    sch = CosineAnnealingLR(opt, T_max=100, eta_min=1e-7)

    data_json1=load_json(args.data_path1)
    data_json2=load_json(args.data_path2)

    aug= {
        "pc_augm_scale": 1.0,
        "pc_augm_rot": False,
        "pc_augm_jitter": False,
        "pc_augm_mirror_prob": 0.0,
    }

    tr_list=data_json1[str(args.fold)]['tr']+data_json2[str(args.fold)]['tr']
    trt_list=data_json1[str(args.fold)]['tr']+data_json2[str(args.fold)]['tr']
    val_list=data_json1[str(args.fold)]['val']+data_json2[str(args.fold)]['val']

    tr_dataset=FaceDataset(tr_list, augmentation=aug, sym=True)
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trt_dataset=FaceDataset(tr_list, sym=True)
    trt_loader = DataLoader(trt_dataset, batch_size=len(tr_list))

    val_dataset=FaceDataset(val_list, sym=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_list))

    CVAR.train()

    start_time = time()
    trt_avg_loss_li=[]
    val_avg_loss_li=[]
    val_avg_mse_li=[]

    loss_fn = torch.nn.CrossEntropyLoss()

    start_epoch=0
    best_loss=  float('inf')

    for epoch in range(start_epoch, args.epochs):

        for data in tr_loader:
            CVAR.train()
            with torch.no_grad():
                complete=data['complete'].to(device).float()                      
                labels_list = MVQVAE.img_to_idxBl(complete)                       
                input_h_list = MVQVAE.idxBl_to_h(labels_list)                     
                x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)             

                cond=data['cond'].to(device).float()                              
                pre_labels_list = MVQVAE.img_to_idxBl(cond)                       
                pre_input_h_list = MVQVAE.idxBl_to_h_cond(pre_labels_list)        
                pre_x_BLCv_wo_first_l = torch.concat(pre_input_h_list, dim=1)     


            logits = CVAR(pre_x_BLCv_wo_first_l, x_BLCv_wo_first_l)
            logits = logits.view(-1, logits.size(-1))

            labels = torch.cat(labels_list, dim=1)
            labels = labels.view(-1)

            loss = loss_fn(logits, labels)

            loss.backward()
            opt.step()
            opt.zero_grad()
        sch.step()

        if (epoch+1)%1==0:
            with torch.no_grad():
                CVAR.eval()   

                trt_loss = 0
                trt_steps = 0
                for data in trt_loader:

                    complete=data['complete'].to(device).float()                      
                    labels_list = MVQVAE.img_to_idxBl(complete)                       
                    input_h_list = MVQVAE.idxBl_to_h(labels_list)                     
                    x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)             

                    cond=data['cond'].to(device).float()                              
                    pre_labels_list = MVQVAE.img_to_idxBl(cond)                       
                    pre_input_h_list = MVQVAE.idxBl_to_h_cond(pre_labels_list)        
                    pre_x_BLCv_wo_first_l = torch.concat(pre_input_h_list, dim=1)     

                    logits = CVAR(pre_x_BLCv_wo_first_l, x_BLCv_wo_first_l)
                    logits = logits.view(-1, logits.size(-1))
                    
                    labels = torch.cat(labels_list, dim=1)
                    labels = labels.view(-1)

                    loss = loss_fn(logits, labels)

                    trt_loss += loss.item()
                    trt_steps +=1

                val_loss = 0
                val_steps = 0
                for data in val_loader:

                    complete=data['complete'].to(device).float()                      
                    labels_list = MVQVAE.img_to_idxBl(complete)                       
                    input_h_list = MVQVAE.idxBl_to_h(labels_list)                     
                    x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)             

                    cond=data['cond'].to(device).float()                              
                    pre_labels_list = MVQVAE.img_to_idxBl(cond)                       
                    pre_input_h_list = MVQVAE.idxBl_to_h_cond(pre_labels_list)        
                    pre_x_BLCv_wo_first_l = torch.concat(pre_input_h_list, dim=1)     

                    logits = CVAR(pre_x_BLCv_wo_first_l, x_BLCv_wo_first_l)
                    logits = logits.view(-1, logits.size(-1))
                    
                    labels = torch.cat(labels_list, dim=1)
                    labels = labels.view(-1)

                    loss = loss_fn(logits, labels)

                    val_loss += loss.item()
                    val_steps +=1


                trt_avg_loss_li.append(trt_loss / trt_steps)
                val_avg_loss_li.append(val_loss / val_steps)

                val_mse = 0
                with torch.no_grad():
                    for data in val_loader:

                        complete=data['complete'].to(device).float()                      
                        labels_list = MVQVAE.img_to_idxBl(complete)                       
                        input_h_list = MVQVAE.idxBl_to_h(labels_list)                     
                        x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)             

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

                        val_ls=F.mse_loss(recon, complete)*1000
                        val_mse += val_ls.item()

                val_avg_mse_li.append(val_mse / val_steps)

                end_time = time()
                sec_per_epoch = (end_time - start_time)

                logger.info(f"Epoch={epoch+1:07d}, Train Loss: {trt_avg_loss_li[-1]:.4f}, Val Loss: {val_avg_loss_li[-1]:.4f}, Val MSE: {val_avg_mse_li[-1]:.4f}, Sec: {sec_per_epoch:.2f}")
                start_time = time()

                if val_avg_mse_li[-1] < best_loss:
                    checkpoint = {
                        "cvar": CVAR.state_dict(),
                        "opt": opt.state_dict(),
                        "sch": sch.state_dict(),
                        "args": args,
                        "tr_loss": trt_avg_loss_li,
                        "val_loss": val_avg_loss_li,
                        "val_mse": val_avg_mse_li,
                    }

                    checkpoint_path = f"{checkpoint_dir}/best.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    best_loss = val_avg_mse_li[-1]

        # Save checkpoint:
        if (epoch+1)%10==0:
            checkpoint = {
                "cvar": CVAR.state_dict(),
                "opt": opt.state_dict(),
                "sch": sch.state_dict(),
                "args": args,
                "tr_loss": trt_avg_loss_li,
                "val_loss": val_avg_loss_li,
                "val_mse": val_avg_mse_li,
            }

            checkpoint_path = f"{checkpoint_dir}/{epoch+1:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    CVAR.eval()

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)

    parser.add_argument("--experiment_dir", type=str)
    parser.add_argument("--mvqvae_path", type=str)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--data_path1", type=str)
    parser.add_argument("--data_path2", type=str)

    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--cvar_model", type=str, default='CVAR-T')
    parser.add_argument("--cvar_depth_cond", type=int, default=10)

    parser.add_argument("--point_size", type=int, default=282)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--mvae_results_dir", type=str)
    parser.add_argument("--mvae_model", type=str, default='VQVAET-T')
    parser.add_argument("--mvae_num_cbembed", type=int, default=512)
    parser.add_argument("--mvae_dim_cbembed", type=int, default=64)
    parser.add_argument("--mvae_dist_typ", type=str, default='geo')
    parser.add_argument("--mvae_scale", type=int, default=8)

    parser.add_argument("--pt_results_dir", type=str)
    parser.add_argument("--pt_model", type=str, default='CVAR-T')

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument('-g', '--gpu', type=int, default=0)

    args = parser.parse_args()
    main(args)
