import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import wandb
import os
import h5py
from transformers import get_cosine_schedule_with_warmup
from DEXA.dexa_fm.lejepa_dataset import LeJEPAHDF5Dataset
from model import LeJEPA_Encoder, SIGReg
from DEXA.dexa_fm.augmentations import train_transforms, val_transforms


config = {
    "img_size": (384, 128),
    "batch_size": 64,
    "lr": 5e-4,
    "probe_lr": 1e-3,
    "weight_decay": 5e-2,
    "epochs": 300,
    "lambda": 0.05,
    "sigreg_slices": 2048,
    "warmup_epochs": 30,
    "global_views": 4,
    "local_views": 8,
    "model_name": 'vit_base_patch16_384',
    "drop_path": 0.1,
}


# NEW PATH CONFIGURATION
HDF5_PATH = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
TARGETS_CSV = "/net/mraid20/export/genie/LabData/Analyses/10K_Trajectories/body_systems/Age_Gender_BMI.csv"
CHECKPOINTS = '/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/'


RESUME_FROM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


run_name = f"updated_normalization"


class OnlineLinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.probe(x)


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
    wandb.init(entity="gilsasson12-weizmann-institute-of-science", project="LeJEPA_DEXA_Scratch", name=run_name, resume="allow")

    print(f"Loading targets from {TARGETS_CSV}...")
    targets_orig = pd.read_csv(TARGETS_CSV)

    # STRICT: Set MultiIndex as requested
    targets_orig.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    targets_orig.sort_index(inplace=True)

    # Normalize Targets
    age_mean = targets_orig['age'].mean()
    age_std = targets_orig['age'].std()
    targets_orig['age'] = (targets_orig['age'] - age_mean) / age_std
    print(f"Target Normalization: Mean={age_mean:.2f}, Std={age_std:.2f}")

    # SCAN HDF5 & GROUP BY SUBJECT (REGISTRATION CODE)
    print(f"Scanning HDF5 keys from {HDF5_PATH}...")

    with h5py.File(HDF5_PATH, 'r') as f:
        all_raw_keys = list(f.keys())

    # Map: { "10K_12345": ["10K_12345_baseline", "10K_12345_visit1"] }
    subject_map = {}

    for key in all_raw_keys:
        parts = key.split('_')
        # Reconstruct Subject ID (RegistrationCode): "10K_12345"
        s_id = "_".join(parts[:2])

        if s_id not in subject_map:
            subject_map[s_id] = []
        subject_map[s_id].append(key)

    unique_subjects = list(subject_map.keys())
    print(f"Total HDF5 images: {len(all_raw_keys)}")
    print(f"Total Unique Patients: {len(unique_subjects)}")

    # IDENTIFY LABELED VS UNLABELED PATIENTS

    # Check which patients exist in your targets CSV (Level 0 check)
    valid_target_ids = set(targets_orig.index.get_level_values(0))

    labeled_subjs = []
    unlabeled_subjs = []

    for s in unique_subjects:
        if s in valid_target_ids:
            labeled_subjs.append(s)
        else:
            unlabeled_subjs.append(s)

    print(f" - Patients with Labels: {len(labeled_subjs)}")
    print(f" - Patients w/o Labels: {len(unlabeled_subjs)} (Adding to Train)")

    # Split patients
    train_subs, val_subs = train_test_split(labeled_subjs, test_size=0.2, random_state=42)

    # Combine: Train = All Unlabeled + 80% Labeled
    final_train_subs = unlabeled_subjs + train_subs
    final_val_subs = val_subs

    # --- Prepare Train Targets ---
    targets_train = targets_orig.copy()

    # Set labels to NaN for all validation subjects in the TRAIN dataframe
    val_mask = targets_train.index.get_level_values(0).isin(final_val_subs)
    targets_train.loc[val_mask, 'age'] = np.nan

    # --- TRAIN KEYS ---
    # Simply grab all images for the training patients
    train_keys = []
    for s in final_train_subs:
        train_keys.extend(subject_map[s])

    # --- VAL KEYS ---
    val_keys = []

    print("Building Validation Set (Strict Match: Recode + Visit)...")
    for s in final_val_subs:
        potential_keys = subject_map[s]
        for k in potential_keys:
            # Re-parse to check specific visit label
            parts = k.split('_')
            s_id = "_".join(parts[:2])

            # STRICT LOGIC from your snippet: join everything after index 2
            v_id = "_".join(parts[2:])

            # Check if this specific visit (e.g. baseline) has a value
            if (s_id, v_id) in targets_orig.index:
                val = targets_orig.loc[(s_id, v_id), 'age']
                if isinstance(val, pd.Series):
                    val = val.iloc[0]

                # Only add if not NaN
                if not pd.isna(val):
                    val_keys.append(k)

    print(f"Final Split Statistics:")
    print(f" - Train Images: {len(train_keys)}")
    print(f" - Val Images: {len(val_keys)}")

    # 3. Instantiate the Transforms
    train_aug = train_transforms(global_size=(384, 128), local_size=(96, 96))
    val_aug = val_transforms(global_size=(384, 128))

    # 4. Create Datasets
    train_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=train_keys,
        targets_df=targets_train,
        transform=train_aug,
        n_global=config["global_views"],
        n_local=config["local_views"]
    )

    # Val dataset uses targets_orig which has the values
    val_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=val_keys,
        targets_df=targets_orig,
        transform=val_aug,
        n_global=2,
        n_local=0
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=lejepa_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=lejepa_collate_fn, pin_memory=True)

    # ---------------------------------------------------------
    # 7. MODEL & LOOP

    encoder = LeJEPA_Encoder(config["model_name"], img_size=config['img_size'], proj_out_dim=64).to(DEVICE)

    # Probe is detached from encoder for SSL monitoring
    probe = nn.Linear(encoder.embed_dim, 1).to(DEVICE)

    sigreg_module = SIGReg(num_slices=config["sigreg_slices"]).to(DEVICE)

    # Optimizer for encoder (SSL loss)
    opt_enc = optim.AdamW(encoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Separate optimizer for probe (detached for SSL monitoring)
    opt_probe = optim.AdamW(probe.parameters(), lr=config["probe_lr"], weight_decay=1e-5)

    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)

    mse_crit = nn.MSELoss()

    start_epoch = 1
    best_val_r2 = -float('inf')
    global_step = 0

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        probe.load_state_dict(checkpoint['probe'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        global_step = checkpoint.get('global_step', 0)  # Restore step counter
        print(f"Resumed from Epoch {start_epoch}, global_step={global_step}")

    print("Starting Training...")

    for epoch in range(start_epoch, config["epochs"] + 1):
        encoder.train()
        probe.train()

        train_stats = {'loss': 0, 'pred': 0, 'sig': 0, 'probe': 0}
        train_preds, train_trues = [], []

        for views, age_targets in train_loader:
            age_targets = age_targets.to(DEVICE).view(-1)
            bs = age_targets.size(0)

            # 1. Forward Pass (SSL) - LeJEPA forward logic
            n_globals = config["global_views"]
            n_locals = config["local_views"]

            global_views = torch.cat(views[:n_globals], dim=0).to(DEVICE)
            g_feats, g_projs = encoder(global_views)

            local_inputs = torch.cat(views[n_globals:], dim=0).to(DEVICE)

            _, l_projs = encoder(local_inputs)

            g_projs = g_projs.view(n_globals, bs, -1)
            l_projs = l_projs.view(n_locals, bs, -1)
            all_projs = torch.cat([g_projs, l_projs], dim=0)

            # SSL Loss (Prediction)
            center = g_projs.mean(dim=0)
            loss_pred = (all_projs - center.unsqueeze(0)).square().mean()

            # SSL Loss (SIGReg)
            loss_sigreg = 0.0
            num_views = all_projs.shape[0] # Should be (n_globals + n_locals)

            for i in range(num_views):
                # Shape: (Batch_Size, Dim)
                view_batch = all_projs[i] 
                loss_sigreg += sigreg_module(view_batch)

            # Average the results
            loss_sigreg = loss_sigreg / num_views

            # Combined SSL Loss
            loss_ssl = (1 - config["lambda"]) * loss_pred + config["lambda"] * loss_sigreg

            opt_enc.zero_grad()
            loss_ssl.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt_enc.step()

            # 2. Probe update with DETACHED features for SSL monitoring
            label_mask = (age_targets != -1.0) & (~torch.isnan(age_targets))
            loss_probe = torch.tensor(0.0, device=DEVICE)

            if label_mask.sum() > 0:
                g_feats_reshaped = g_feats.view(n_globals, bs, -1)
                feat_pooled = g_feats_reshaped.mean(dim=0).detach()  # DETACHED: probe doesn't affect encoder

                feat_labeled = feat_pooled[label_mask]
                target_labeled = age_targets[label_mask]

                age_pred = probe(feat_labeled).view(-1)
                loss_probe = mse_crit(age_pred, target_labeled)

                opt_probe.zero_grad()
                loss_probe.backward()
                opt_probe.step()

                train_preds.extend(age_pred.detach().cpu().numpy())
                train_trues.extend(target_labeled.cpu().numpy())

            scheduler.step()

            train_stats['loss'] += loss_ssl.item()
            train_stats['pred'] += loss_pred.item()
            train_stats['sig'] += loss_sigreg.item()
            train_stats['probe'] += loss_probe.item()

            global_step += 1

        train_r2 = r2_score(train_trues, train_preds) if len(train_trues) > 1 else 0.0

        # --- VALIDATION LOOP ---
        encoder.eval()
        probe.eval()
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
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(age_targets.cpu().numpy())

        val_r2 = r2_score(val_trues, val_preds) if len(val_trues) > 0 else 0.0

        wandb.log({
            "epoch": epoch,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_loss": train_stats['loss'] / len(train_loader),
            "train_pred_loss": train_stats['pred'] / len(train_loader),
            "train_sig_loss": train_stats['sig'] / len(train_loader),
            "lr": scheduler.get_last_lr()[0]
        }, step=global_step)

        print(f"Ep {epoch} | Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f}")

        if val_r2 > best_val_r2 or epoch % 50 == 0:
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'probe': probe.state_dict(),
                'best_val_r2': best_val_r2,
                'global_step': global_step  # Save step counter
            }, os.path.join(CHECKPOINTS, f'best_model_{run_name}.pth'))


if __name__ == "__main__":
    main()
