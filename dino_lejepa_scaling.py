import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import wandb
import os
from transformers import get_cosine_schedule_with_warmup
# Ensure these files are in your python path
from lejepa_dataset import LeJEPADEXADataset
from model import LeJEPA_Encoder, SIGReg
from helpers import *
from augmentations import LeJEPATransformConfig

# --- CONFIGURATION ---
default_config = {
    "batch_size": 64,
    "lr": 1e-4,
    "probe_lr": 1e-3,
    "weight_decay": 5e-2,
    "epochs": 200,
    "lambda": 0.2,
    "sigreg_slices": 4096,
    "warmup_epochs": 20,
    "global_views": 2,
    "local_views": 8,
    "model_name": 'vit_small_patch16_224', #
    "drop_path": 0.1
}

# --- PATHS ---
FULL_BODY_MANIFEST = '/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl'
CROPS_MANIFEST = '/net/mraid20/export/genie/LabData/Analyses/gilsa/crops_manifest.pkl'
TARGETS_CSV = '/home/gilsa/PycharmProjects/DEXA/targets_for_downstream_augmented.csv'
CHECKPOINTS_DIR = '/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_scaling/'
DINO_PKL_PATH = '/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/dexa_zeroshot.pkl'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def evaluate_dino_from_pickle(pkl_path, train_ids, val_ids, targets_df):
    """Computes the DINO Baseline R2 Score using pre-extracted embeddings."""
    print(f"\n[DINO] Loading Embeddings from {pkl_path}...")
    try:
        embeddings = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"[DINO] Error loading pickle: {e}")
        return 0.0

    print(embeddings.head())
    embeddings.set_index(['RegistrationCode', 'research_stage'], inplace=True)

    # Align Indices
    common_indices = embeddings.index.intersection(targets_df.index)
    print(f"[DINO] Found {len(common_indices)} matching samples.")

    X = embeddings.loc[common_indices].values
    y = targets_df.loc[common_indices, 'age'].values

    # Identify Train/Val based on Patient ID (RegistrationCode)
    patient_ids = common_indices.get_level_values('RegistrationCode')

    train_mask = patient_ids.isin(train_ids)
    val_mask = patient_ids.isin(val_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    score = r2_score(y_val, preds)

    print(f"[DINO] Baseline R2: {score:.4f}")
    return score

# --- MAIN EXPERIMENT ---

def main():
    wandb.init(entity="gilsasson12-weizmann-institute-of-science", project="LeJEPA_Scaling_Graph", name="Scaling_Experiment_v1")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # 1. PREPARE DATA
    manifest = prepare_combined_manifest(FULL_BODY_MANIFEST, CROPS_MANIFEST)
    targets_orig = pd.read_csv(TARGETS_CSV, index_col=[0, 1])
    targets_orig.sort_index(inplace=True)

    # Normalize Targets
    age_mean = targets_orig['age'].mean()
    age_std = targets_orig['age'].std()
    targets_orig['age'] = (targets_orig['age'] - age_mean) / age_std

    # Identify valid splits
    valid_keys = set(targets_orig.index)
    manifest['has_label'] = manifest.apply(lambda r: (r['RegistrationCode'], r['research_stage']) in valid_keys, axis=1)
    labeled_patients = manifest[manifest['has_label'] == True]['RegistrationCode'].unique()

    # MASTER SPLIT: Fixed Validation Set (20% of patients)
    train_ids_master, val_ids_master = train_test_split(labeled_patients, test_size=0.2, random_state=42)
    print(f"Master Split | Train Patients: {len(train_ids_master)} | Val Patients: {len(val_ids_master)}")

    # 2. DINO BASELINE (Computed Once)
    dino_r2 = evaluate_dino_from_pickle(DINO_PKL_PATH, train_ids_master, val_ids_master, targets_orig)

    # 3. FIXED VALIDATION LOADER (Used for all LeJEPA runs)
    val_manifest = manifest[manifest['RegistrationCode'].isin(val_ids_master)]
    val_manifest = val_manifest[val_manifest['has_label'] == True].reset_index(drop=True)
    val_dataset = LeJEPADEXADataset(val_manifest, targets_orig, LeJEPATransformConfig(), n_global=10, n_local=0)

    val_loader = DataLoader(
        val_dataset, batch_size=default_config["batch_size"],
        shuffle=False, num_workers=4, collate_fn=lejepa_collate_fn
    )

    # 4. SCALING EXPERIMENT LOOP
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0] # 10% to 100% Data
    results = {}

    for ratio in ratios:
        # Re-initialize WandB run for cleanliness or use step logic
        print(f"\n>>> STARTING TRAINING :: DATA RATIO {ratio*100}% <<<")

        # A. Subset Training IDs
        if ratio < 1.0:
            subset_size = int(len(train_ids_master) * ratio)
            np.random.seed(42) # Ensure 25% contains the 10%
            train_ids_subset = np.random.choice(train_ids_master, subset_size, replace=False)
        else:
            train_ids_subset = train_ids_master

        # B. Create Train Loader
        train_manifest_subset = manifest[manifest['RegistrationCode'].isin(train_ids_subset)]

        # Re-index targets for subset (Probe only sees these labels)
        manifest_keys = pd.MultiIndex.from_frame(train_manifest_subset[['RegistrationCode', 'research_stage']])
        targets_train_subset = targets_orig.reindex(manifest_keys)

        train_dataset = LeJEPADEXADataset(
            train_manifest_subset, targets_train_subset, LeJEPATransformConfig(),
            n_global=default_config["global_views"], n_local=default_config["local_views"]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=default_config["batch_size"],
            shuffle=True, num_workers=4, collate_fn=lejepa_collate_fn
        )

        print(f"Subset Size: {len(train_dataset)} images ({len(train_ids_subset)} patients)")

        # C. RESET MODEL (Start From Scratch)
        encoder = LeJEPA_Encoder(default_config["model_name"], proj_out_dim=64, pretrained=False, drop_path_rate=default_config['drop_path']).to(DEVICE)
        probe = OnlineLinearProbe(encoder.embed_dim).to(DEVICE)
        sigreg_module = SIGReg(num_slices=default_config["sigreg_slices"]).to(DEVICE)

        opt_enc = optim.AdamW(encoder.parameters(), lr=default_config["lr"], weight_decay=default_config["weight_decay"])
        opt_probe = optim.AdamW(probe.parameters(), lr=default_config["probe_lr"], weight_decay=1e-3)

        total_steps = default_config["epochs"] * len(train_loader)
        warmup_steps = default_config["warmup_epochs"] * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)
        mse_crit = nn.MSELoss()

        best_val_r2 = -float('inf')

        # D. TRAINING EPOCHS
        for epoch in range(1, default_config["epochs"] + 1):
            encoder.train(); probe.train()
            train_preds, train_trues = [], []
            train_loss = 0

            for views, age_targets in train_loader:
                # -- Forward Pass --
                n_g, n_l = default_config["global_views"], default_config["local_views"]

                global_inputs = torch.cat(views[:n_g], dim=0).to(DEVICE)
                local_inputs = torch.cat(views[n_g:], dim=0).to(DEVICE)

                g_feats, g_projs = encoder(global_inputs)
                _, l_projs = encoder(local_inputs)

                bs = age_targets.size(0)
                all_views = torch.cat([g_projs.view(n_g, bs, -1), l_projs.view(n_l, bs, -1)], dim=0)

                # JEPALoss
                loss_pred = (all_views - all_views.mean(dim=0).unsqueeze(0)).square().mean()
                loss_sigreg = sigreg_module(all_views.reshape(-1, all_views.shape[-1]))
                loss_total = (1 - default_config["lambda"]) * loss_pred + default_config["lambda"] * loss_sigreg

                opt_enc.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt_enc.step()

                # -- Probe Update --
                age_targets = age_targets.to(DEVICE).view(-1)
                label_mask = ~torch.isnan(age_targets)
                if label_mask.sum() > 0:
                    feat_pooled = g_feats.view(n_g, bs, -1).mean(dim=0).detach()
                    pred = probe(feat_pooled[label_mask]).squeeze()
                    loss_p = mse_crit(pred, age_targets[label_mask])

                    opt_probe.zero_grad()
                    loss_p.backward()
                    opt_probe.step()

                    train_preds.extend(pred.detach().cpu().numpy())
                    train_trues.extend(age_targets[label_mask].cpu().numpy())

                scheduler.step()
                train_loss += loss_total.item()

            # E. VALIDATION (Fixed Set)
            encoder.eval(); probe.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for views, age_targets in val_loader:
                    # Stack & Flatten Global Views
                    g_views = torch.stack(views).to(DEVICE) # (10, B, 3, 224, 224)
                    n_v, b, c, h, w = g_views.shape

                    feats, _ = encoder(g_views.view(-1, c, h, w))
                    feats_avg = feats.view(n_v, b, -1).mean(dim=0)

                    pred = probe(feats_avg).squeeze()
                    val_preds.extend(pred.cpu().numpy())
                    val_trues.extend(age_targets.view(-1).cpu().numpy())

            val_r2 = r2_score(val_trues, val_preds)

            # Log Metrics: Note 'ratio' allows grouping in WandB
            wandb.log({
                "epoch": epoch,
                "data_ratio": ratio,
                "val_r2": val_r2,
                "dino_baseline": dino_r2, # Constant Line
                "train_loss": train_loss / len(train_loader)
            })

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                # Save checkpoint specific to this ratio
                torch.save({
                    'ratio': ratio,
                    'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'probe': probe.state_dict(),
                }, os.path.join(CHECKPOINTS_DIR, f'best_model_ratio_{ratio}.pth'))

            print(f"Ratio {ratio} | Ep {epoch} | Val R2: {val_r2:.4f} (Best: {best_val_r2:.4f})")

        results[ratio] = best_val_r2

    print("\n=== FINAL SCALING RESULTS ===")
    print(f"DINO Baseline: {dino_r2}")
    print(f"LeJEPA Results: {results}")

if __name__ == "__main__":
    main()