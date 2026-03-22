import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import wandb
import os
import h5py
from transformers import get_cosine_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from DEXA.dexa_fm.distributed_utils import setup_ddp, cleanup_ddp, is_main_process
from DEXA.dexa_fm.lejepa_dataset import LeJEPAHDF5Dataset
from DEXA.dexa_fm.model import LeJEPA_Encoder, SIGReg
from DEXA.dexa_fm.augmentations import train_transforms, val_transforms
from sklearn.model_selection import train_test_split
from DEXA.dexa_fm.helpers import *
import random

UKBB_MEAN = [0.18899241089820862, 0.18899241089820862, 0.18899241089820862]
UKBB_STD  = [0.2798040509223938, 0.2798040509223938, 0.2798040509223938]

config = {
    "img_size": (384, 128),
    "batch_size": 128, # PER GPU
    "lr": 5e-4,
    "probe_lr": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 400,
    "lambda": 0.05,
    "sigreg_slices": 2048,
    "warmup_epochs": 15,
    "global_views": 2,
    "local_views": 8,
    "model_name": 'vit_base_patch16_384',
    "drop_path": 0.1,
    "subset_fraction": 1,
    "workers": 6
}

from LabData.DataLoaders import UkbbLoader
ukb_loader = UkbbLoader.UkbbLoader()

HDF5_PATH = '/net/mraid20/export/jasmine/UKBB_DATA/ukbb_dexa_dataset_v3.h5'
CHECKPOINTS = os.getenv('CHECKPOINTS_DIR', '/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/')

run_name = os.getenv('RUN_NAME')
RESUME_FROM =  os.path.join(CHECKPOINTS, f"{run_name}.pth")

class OnlineLinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)
    def forward(self, x): return self.probe(x)

def lejepa_collate_fn(batch):
    views_lists, targets = zip(*batch)
    collated_targets = torch.stack(targets)
    num_views = len(views_lists[0])
    collated_views = []
    for i in range(num_views):
        view_batch = [patient_views[i] for patient_views in views_lists]
        collated_views.append(torch.stack(view_batch))
    return collated_views, collated_targets

def main():
     # --- CLEAN MULTI-GPU SETUP ---
    local_rank, global_rank, DEVICE = setup_ddp()

    # ONLY Rank 0 initializes WandB now!
    if is_main_process():
        wandb.init(
            entity="gilsasson12-weizmann-institute-of-science", 
            project="LeJEPA_DEXA_Scratch", 
            name=run_name,
            resume="allow",
            id=run_name
        )

    # --- DATA PREPARATION ---
    targets_wide = ukb_loader.load_ukbb(by='field_name', which=['Age when attended assessment centre'], instances=[2, 3])
    
    if 'eid - visit 2' in targets_wide.columns:
        targets_wide['eid'] = targets_wide['eid - visit 2']
    elif 'eid' in targets_wide.index.names or targets_wide.index.name == 'eid':
        targets_wide.reset_index(inplace=True)
    
    col_v2 = 'Age when attended assessment centre - visit 2'
    col_v3 = 'Age when attended assessment centre - visit 3'

    df_v2 = targets_wide[['eid', col_v2]].dropna().copy()
    df_v2.columns = ['eid', 'age']
    df_v2['visit'] = '2'

    df_v3 = targets_wide[['eid', col_v3]].dropna().copy()
    df_v3.columns = ['eid', 'age']
    df_v3['visit'] = '3'

    targets_orig = pd.concat([df_v2, df_v3])
    targets_orig['eid'] = targets_orig['eid'].apply(lambda x: str(x).split('.')[0].strip())
    targets_orig.set_index(['eid', 'visit'], inplace=True)
    targets_orig.index.names = ['RegistrationCode', 'research_stage']

    targets_orig, mu, sigma = normalize_targets(targets_orig, 'age')
    
    if is_main_process():
        print(f"Loaded {len(targets_orig)} valid age targets across Visit 2 and Visit 3.")
        print(f"Scanning HDF5 keys from {HDF5_PATH}...")
        
    with h5py.File(HDF5_PATH, 'r') as f:
        all_raw_keys = list(f.keys())

    subject_map = {}
    for key in all_raw_keys:
        parts = key.split('_')
        s_id = str(parts[0]).split('.')[0].strip() 
        if s_id not in subject_map:
            subject_map[s_id] = []
        subject_map[s_id].append(key)

    unique_subjects = list(subject_map.keys())

    if config.get("subset_fraction", 1.0) < 1.0:
        fraction = config["subset_fraction"]
        random.seed(37)
        random.shuffle(unique_subjects)
        unique_subjects = unique_subjects[:int(len(unique_subjects) * fraction)]

    labeled_subjs, unlabeled_subjs = [], []
    valid_target_ids = set(str(x) for x in targets_orig.index.get_level_values('RegistrationCode'))

    for s in unique_subjects:
        if str(s) in valid_target_ids: 
            labeled_subjs.append(s)
        else: 
            unlabeled_subjs.append(s)
            
    if is_main_process() and len(labeled_subjs) == 0:
        raise ValueError("Critical Match Failure: Still got 0 labeled patients.")

    train_subs, val_subs = train_test_split(labeled_subjs, test_size=0.2, random_state=42)
    final_train_subs = unlabeled_subjs + train_subs
    final_val_subs = val_subs

    targets_train = targets_orig.copy()
    val_subs_set = set(final_val_subs) 
    mask = targets_train.index.get_level_values('RegistrationCode').isin(val_subs_set)
    targets_train.loc[mask, 'age'] = np.nan

    train_keys = []
    for s in final_train_subs: train_keys.extend(subject_map[s])

    val_keys = []
    for s in final_val_subs:
        for k in subject_map[s]:
            parts = k.split('_')
            s_id, v_id = parts[0], parts[1] 
            if (s_id, v_id) in targets_orig.index:
                val = targets_orig.loc[(s_id, v_id), 'age']
                if isinstance(val, pd.Series): val = val.iloc[0]
                if not pd.isna(val):
                    val_keys.append(k)

    if is_main_process():
        print(f"Final Split Statistics:\n  - Train Images: {len(train_keys)}\n  - Val Images:   {len(val_keys)}")

    train_aug = train_transforms(global_size=(384, 128), local_size=(96, 96), mean=UKBB_MEAN, std=UKBB_STD)
    val_aug   = val_transforms(global_size=(384, 128), mean=UKBB_MEAN, std=UKBB_STD)

    train_dataset = LeJEPAHDF5Dataset(hdf5_path=HDF5_PATH, keys=train_keys, targets_df=targets_train, transform=train_aug, n_global=config["global_views"], n_local=config["local_views"])

    # --- 3. DDP SAMPLER ---
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, sampler=train_sampler, num_workers=config["workers"], collate_fn=lejepa_collate_fn, pin_memory=True)

    if is_main_process():
        val_dataset = LeJEPAHDF5Dataset(hdf5_path=HDF5_PATH, keys=val_keys, targets_df=targets_orig, transform=val_aug, n_global=config["global_views"], n_local=0)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], collate_fn=lejepa_collate_fn, pin_memory=True)

    # --- 4. MODEL DDP WRAPPING ---
    print(f"[Rank {global_rank}] Creating encoder...", flush=True)
    
    # Restored: Actually creating the model and moving it to the GPU
    encoder = LeJEPA_Encoder(config["model_name"], img_size=config['img_size'], proj_out_dim=64, pretrained=False, drop_path_rate=config['drop_path'])
    encoder = encoder.to(DEVICE)

    import torch.distributed as dist
    print(f"[Rank {global_rank}] Waiting at barrier before DDP init...", flush=True)
    dist.barrier(device_ids=[local_rank]) 
    print(f"[Rank {global_rank}] Passed barrier, starting DDP wrapping...", flush=True)

    # Wrapping the encoder in DDP
    encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    # Setting up the probe
    probe = OnlineLinearProbe(encoder.module.embed_dim).to(DEVICE)
    probe = DDP(probe, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    
    sigreg_module = SIGReg(num_slices=config["sigreg_slices"]).to(DEVICE)

    opt_enc = optim.AdamW(encoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    opt_probe = optim.AdamW(probe.parameters(), lr=config["probe_lr"], weight_decay=1e-6)

    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)
    mse_crit = nn.MSELoss()

    start_epoch = 1
    best_val_r2 = -float('inf')
    global_step = 0

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        encoder.module.load_state_dict(checkpoint['encoder']) 
        probe.module.load_state_dict(checkpoint['probe'])     
        start_epoch = checkpoint['epoch'] + 1
        best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        global_step = checkpoint.get('global_step', 0)
        
        # Restore scheduler state to maintain LR continuity
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            if is_main_process():
                print(f"Restored scheduler state (LR: {scheduler.get_last_lr()[0]:.2e})")
        
        if is_main_process():
            print(f"Resumed from epoch {checkpoint['epoch']}, global_step={global_step}")
            # If wandb run exists and has logged steps beyond what we recovered,
            # jump to the next unlogged epoch to avoid backwards step conflict
            if wandb.run and wandb.run.step > global_step:
                steps_per_epoch_approx = len(train_loader)
                extra_epochs = (wandb.run.step - global_step) // steps_per_epoch_approx
                if extra_epochs > 0:
                    print(f"⚠️  WandB has {wandb.run.step} steps logged (beyond checkpoint). Jumping {extra_epochs} epochs ahead.")
                    start_epoch = checkpoint['epoch'] + 1 + extra_epochs
                    global_step = wandb.run.step

    for epoch in range(start_epoch, config["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        
        encoder.train(); probe.train()
        train_stats = {'loss': 0, 'pred': 0, 'sig': 0, 'probe': 0}

        for i, (views, age_targets) in enumerate(train_loader):
            n_globals, n_locals = config["global_views"], config["local_views"]

            global_inputs = torch.cat(views[:n_globals], dim=0).to(DEVICE)
            local_inputs = torch.cat(views[n_globals:], dim=0).to(DEVICE)
            bs = age_targets.size(0)

            opt_enc.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                g_feats, g_projs = encoder(global_inputs)
                _, l_projs = encoder(local_inputs)

                g_projs = g_projs.view(n_globals, bs, -1)
                l_projs = l_projs.view(n_locals, bs, -1)
                all_views = torch.cat([g_projs, l_projs], dim=0)

                center = g_projs.mean(dim=0)
                loss_pred = (all_views - center.unsqueeze(0)).square().mean()

                # SIGReg computed per-view and averaged
                loss_sigreg = 0.0
                num_views = all_views.shape[0]  # (n_globals + n_locals)
                for i in range(num_views):
                    loss_sigreg += sigreg_module(all_views[i])  # (bs, feature_dim)
                loss_sigreg = loss_sigreg / num_views

                loss_total = (1 - config["lambda"]) * loss_pred + config["lambda"] * loss_sigreg

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            
            opt_enc.step()
            scheduler.step()

            # --- PROBE TRAINING (Monitoring Only, Detached Features) ---
            age_targets = age_targets.to(DEVICE).view(-1)
            label_mask = (age_targets != -1.0) & (~torch.isnan(age_targets))
            loss_probe = torch.tensor(0.0, device=DEVICE)

            if label_mask.sum() > 0:
                opt_probe.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    g_feats_reshaped = g_feats.view(n_globals, bs, -1)
                    feat_pooled = g_feats_reshaped.mean(dim=0).detach()  # Detached for monitoring only
                    feat_labeled = feat_pooled[label_mask]
                    target_labeled = age_targets[label_mask]

                    age_pred = probe(feat_labeled).view(-1)
                    loss_probe = mse_crit(age_pred, target_labeled)

                loss_probe.backward()
                opt_probe.step()

            train_stats['loss'] += loss_total.item()
            train_stats['pred'] += loss_pred.item()
            train_stats['sig'] += loss_sigreg.item()
            train_stats['probe'] += loss_probe.item()
            global_step += 1

        # --- 5. ISOLATED VALIDATION (Rank 0 Only) ---
        if is_main_process():
            encoder.eval(); probe.eval()
            val_preds, val_trues = [], []

            with torch.no_grad():
                for views, age_targets in val_loader:
                    global_views = torch.stack(views).to(DEVICE) 
                    n_v, bs, c, h, w = global_views.shape
                    input_flat = global_views.view(-1, c, h, w)
                    feats, _ = encoder(input_flat)
                    feats = feats.view(n_v, bs, -1)
                    feats_avg = feats.mean(dim=0)

                    pred = probe(feats_avg).squeeze()
                    age_targets = age_targets.to(DEVICE).view(-1)
                    
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)
                    
                    val_preds.extend(pred.cpu().numpy())
                    val_trues.extend(age_targets.cpu().numpy())

            val_r2 = r2_score(val_trues, val_preds) if len(val_trues) > 0 else 0.0

            wandb.log({
                "epoch": epoch,
                "val_r2": val_r2,
                "train_loss": train_stats['loss'] / len(train_loader),
                "train_pred_loss": train_stats['pred'] / len(train_loader),
                "train_sig_loss": train_stats['sig'] / len(train_loader),
                "train_probe_loss": train_stats['probe'] / len(train_loader),
                "lr": scheduler.get_last_lr()[0]
            }, step=global_step)

            print(f"Ep {epoch} | Val R2: {val_r2:.4f} | Loss: {train_stats['loss'] / len(train_loader):.4f} | Pred: {train_stats['pred'] / len(train_loader):.4f} | Sig: {train_stats['sig'] / len(train_loader):.4f}")

            # Track best validation R2
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2

            # Save checkpoints every 10 epochs to minimize lost progress
            if epoch % 10 == 0:
                torch.save({
                    'encoder': encoder.module.state_dict(), 
                    'probe': probe.module.state_dict(),
                    'scheduler': scheduler.state_dict(),  # Save scheduler state
                    'epoch': epoch,
                    'best_val_r2': best_val_r2,
                    'val_r2': val_r2,
                    'global_step': global_step
                }, os.path.join(CHECKPOINTS, f"{run_name}.pth"))

        # Wait for Rank 0 to finish validating before moving to the next epoch
        import torch.distributed as dist
        dist.barrier(device_ids=[local_rank])

    # --- CLEANUP ---
    cleanup_ddp()

if __name__ == "__main__":
    main()
