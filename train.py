import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm
from sklearn.metrics import r2_score
import wandb
import os
from transformers import get_cosine_schedule_with_warmup
from lejepa_dataset import LeJEPADEXADataset
from model import LeJEPA_Encoder
from helpers import *

default_config = {
    "batch_size": 64,
    "lr": 1e-3,
    "probe_lr": 1e-3,
    "weight_decay": 5e-2,
    "epochs": 200,
    "lambda": 0.2,
    "sigreg_slices": 2048,
    "warmup_epochs": 20,
    "global_views": 2,
    "local_views": 8,
    "model_name": 'convnext_small',
}

# PATHS
FULL_BODY_MANIFEST = '/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl'
CROPS_MANIFEST = '/net/mraid20/export/genie/LabData/Analyses/gilsa/crops_manifest.pkl'
TARGETS_CSV = '/home/gilsa/PycharmProjects/DEXA/targets_for_downstream_augmented.csv'
CHECKPOINTS = '/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/dexa_fm/'

# STATS
DEXA_MEAN = [0.12394659966230392, 0.2626885771751404, 0.3075794577598572]
DEXA_STD  = [0.21822425723075867, 0.31778785586357117, 0.3350508213043213]

RESUME_FROM = None
MODEL = 'vit_small_patch16_224'
IS_PRETRAINED = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

run_name = f"LeJEPA_DEXA_{MODEL}_lr{default_config['lr']}lambda{default_config['lambda']}"

class LeJEPATransformConfig:
    """Holds the transformation pipelines"""
    def __init__(self, global_size=224, local_size=96):
        self.normalize = transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD)
        self.to_float = transforms.ConvertImageDtype(torch.float)

        self.fb_pre = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.intensity_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2)

        # B. Global Transform
        self.global_trans = transforms.Compose([
            self.to_float,
            transforms.RandomResizedCrop(global_size, scale=(0.3, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, .1)], p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])

        # C. Local Transform
        self.local_trans = transforms.Compose([
            self.to_float,
            transforms.RandomResizedCrop(local_size, scale=(0.25, 1.0), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])

        # D. Synthetic Local Source Transform
        self.synthetic_local_trans = transforms.Compose([
            self.to_float,
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.3), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])

# Models & Loss
class SIGReg(nn.Module):
    def __init__(self, num_slices=2048, knots=17, integration_limit=5):
        super().__init__()
        self.num_slices = num_slices

        # Pre-calculate integration points (The "Grid")
        t = torch.linspace(-integration_limit, integration_limit, knots, dtype=torch.float32)
        dt = (2 * integration_limit) / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # Gaussian target function (The "Ideal Shape")
        target_cf = torch.exp(-0.5 * t.square())

        # Register as buffers so they move to GPU automatically but aren't trained
        self.register_buffer("t", t.view(1, 1, -1)) # Shape for broadcasting
        self.register_buffer("weights", weights)
        self.register_buffer("target_cf", target_cf.view(1, -1))

    def forward(self, z):
        """
        z: (Batch_Size * Views, Dim)
        """
        B, D = z.shape

        # Generate Random Projections (The "Slices")
        A = torch.randn(D, self.num_slices, device=z.device)
        A = A.div_(A.norm(dim=0, keepdim=True) + 1e-6)

        # Project Data -> (Batch, Slices)
        z_proj = z @ A

        # Compute Empirical Characteristic Function (ECF)
        # val: (Batch, Slices, Knots)
        val = z_proj.unsqueeze(-1) * self.t

        # ecf: (Slices, Knots) -> Average over Batch
        ecf_real = val.cos().mean(dim=0)
        ecf_imag = val.sin().mean(dim=0)

        # Compute Weighted Error
        diff = (ecf_real - self.target_cf).square() + ecf_imag.square()

        # Integrate error (Trapezoidal rule)
        loss_per_slice = diff @ self.weights

        # Return Mean over slices
        return loss_per_slice.mean()


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
    wandb.init(entity="gilsasson12-weizmann-institute-of-science", project="LeJEPA_DEXA_Scratch", name=run_name, resume="allow")

    # Load Everything
    manifest = prepare_combined_manifest(FULL_BODY_MANIFEST, CROPS_MANIFEST)

    # HANDLE DUPLICATES & LOAD TARGETS
    targets_orig = pd.read_csv(TARGETS_CSV, index_col=[0, 1])
    targets_orig.sort_index(inplace=True)

    # Normalize Targets
    age_mean = targets_orig['age'].mean()
    age_std = targets_orig['age'].std()
    targets_orig['age'] = (targets_orig['age'] - age_mean) / age_std

    # Identify Splits (Patient Level)
    valid_keys = set(targets_orig.index)

    # "Has Label" means this SPECIFIC visit has a label
    manifest['has_label'] = manifest.apply(lambda r: (r['RegistrationCode'], r['research_stage']) in valid_keys, axis=1)

    # Get patients who have AT LEAST ONE labeled visit
    labeled_patients = manifest[manifest['has_label'] == True]['RegistrationCode'].unique()

    train_ids, val_ids = train_test_split(labeled_patients, test_size=0.2, random_state=42)

    # Setup Training Targets
    # Align targets with the full manifest (missing visits -> NaN)
    manifest_keys = pd.MultiIndex.from_frame(manifest[['RegistrationCode', 'research_stage']])
    targets_train_masked = targets_orig.reindex(manifest_keys)

    # Mask out validation patients in the training targets
    val_mask = targets_train_masked.index.get_level_values(0).isin(val_ids)
    targets_train_masked.loc[val_mask, 'age'] = np.nan

    # Setup Datasets

    # TRAIN: Uses ALL manifest rows. Probe learns from non-NaN entries.
    train_config = LeJEPATransformConfig()

    # Directly use config keys here
    train_dataset = LeJEPADEXADataset(
        manifest,
        targets_train_masked,
        train_config,
        n_global=default_config["global_views"],
        n_local=default_config["local_views"]
    )

    # We only want to validate on rows that actually have labels.
    # Select rows belonging to Validation IDs
    val_manifest = manifest[manifest['RegistrationCode'].isin(val_ids)]

    #  Only keep rows where 'has_label' is True
    # This prevents the KeyError by removing visits that don't exist in targets_orig
    val_manifest = val_manifest[val_manifest['has_label'] == True].reset_index(drop=True)

    val_dataset = LeJEPADEXADataset(val_manifest, targets_orig, train_config, n_global=10, n_local=0)

    print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    # NOTE: Use same collate_fn for both
    # BATCH_SIZE -> default_config["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=default_config["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=lejepa_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=default_config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=lejepa_collate_fn
    )

    # Model Setup
    encoder = LeJEPA_Encoder(default_config["model_name"], proj_out_dim=64, pretrained=IS_PRETRAINED).to(DEVICE)
    probe = OnlineLinearProbe(encoder.embed_dim).to(DEVICE)

    # SIGREG_SLICES -> default_config["sigreg_slices"]
    sigreg_module = SIGReg(num_slices=default_config["sigreg_slices"]).to(DEVICE)

    opt_enc = optim.AdamW(encoder.parameters(), lr=default_config["lr"], weight_decay=default_config["weight_decay"])

    opt_probe = optim.AdamW(probe.parameters(), lr=default_config["probe_lr"], weight_decay=1e-3)

    total_steps = default_config["epochs"] * len(train_loader)

    warmup_steps = default_config["warmup_epochs"] * len(train_loader)

    scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)
    mse_crit = nn.MSELoss()

    start_epoch = 1
    best_val_r2 = -float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        probe.load_state_dict(checkpoint['probe'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        print(f"Resumed from Epoch {start_epoch}")

    print("Starting Training...")
    global_step = 0

    for epoch in range(start_epoch, default_config["epochs"] + 1):
        encoder.train()
        probe.train()
        train_stats = {'loss': 0, 'pred': 0, 'sig': 0, 'probe': 0}
        train_preds, train_trues = [], []

        # TRAINING LOOP
        for i, (views, age_targets) in enumerate(train_loader):

            # --- Encoder Forward ---
            n_globals = default_config["global_views"]
            n_locals = default_config["local_views"]

            # Global Views
            global_inputs = torch.cat(views[:n_globals], dim=0).to(DEVICE)
            g_feats, g_projs = encoder(global_inputs)

            # Local Views
            local_inputs = torch.cat(views[n_globals:], dim=0).to(DEVICE)
            _, l_projs = encoder(local_inputs)

            bs = age_targets.size(0)
            g_projs = g_projs.view(n_globals, bs, -1)
            l_projs = l_projs.view(n_locals, bs, -1)
            all_views = torch.cat([g_projs, l_projs], dim=0)

            # Losses
            center = all_views.mean(dim=0)
            loss_pred = (all_views - center.unsqueeze(0)).square().mean()

            all_views_flat = all_views.reshape(-1, all_views.shape[-1])
            loss_sigreg = sigreg_module(all_views_flat)

            # LAMBDA -> default_config["lambda_val"]
            loss_total = (1 - default_config["lambda"]) * loss_pred + default_config["lambda"] * loss_sigreg

            opt_enc.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            opt_enc.step()

            # Probe Forward (Masked)
            age_targets = age_targets.to(DEVICE).view(-1)
            label_mask = ~torch.isnan(age_targets)

            loss_probe = torch.tensor(0.0, device=DEVICE)

            if label_mask.sum() > 0:
                g_feats_reshaped = g_feats.view(n_globals, bs, -1)
                feat_pooled = g_feats_reshaped.mean(dim=0).detach()

                feat_labeled = feat_pooled[label_mask]
                target_labeled = age_targets[label_mask]

                age_pred = probe(feat_labeled).squeeze()
                loss_probe = mse_crit(age_pred, target_labeled)

                opt_probe.zero_grad()
                loss_probe.backward()
                opt_probe.step()

                train_preds.extend(age_pred.detach().cpu().numpy())
                train_trues.extend(target_labeled.cpu().numpy())

            scheduler.step()

            train_stats['loss'] += loss_total.item()
            train_stats['pred'] += loss_pred.item()
            train_stats['sig'] += loss_sigreg.item()
            train_stats['probe'] += loss_probe.item()
            global_step += 1

        train_r2 = r2_score(train_trues, train_preds) if len(train_trues) > 1 else 0.0

        # VALIDATION LOOP
        encoder.eval(); probe.eval()
        val_preds, val_trues = [], []

        with torch.no_grad():
            for views, age_targets in val_loader:
                # views is a list of [View1, View2, ..., View10]
                # We want to AVERAGE their features to get a stable score

                # Stack global views: (N_Global, Batch, 3, 224, 224)
                # We used n_global=10 for val_dataset, so take them all
                global_views = torch.stack(views).to(DEVICE)
                n_v, bs, c, h, w = global_views.shape

                # Flatten -> Encode -> Reshape
                input_flat = global_views.view(-1, c, h, w)

                # Get Features (ignore projections)
                feats, _ = encoder(input_flat)

                # Reshape: (N_Global, Batch, Dim)
                feats = feats.view(n_v, bs, -1)

                # AVERAGE Features (Test-Time Augmentation)
                feats_avg = feats.mean(dim=0) # (Batch, Dim)

                # Predict Age
                pred = probe(feats_avg).squeeze()

                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(age_targets.view(-1).cpu().numpy())

        val_r2 = r2_score(val_trues, val_preds)

        wandb.log({
            "epoch": epoch,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_loss": train_stats['loss'] / len(train_loader),
            "lr": scheduler.get_last_lr()[0]
        }, step=global_step)

        print(f"Ep {epoch} | Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'probe': probe.state_dict(),
                'best_val_r2': best_val_r2
            }, os.path.join(CHECKPOINTS, f'best_model_{run_name}.pth'))


if __name__ == "__main__":
    main()