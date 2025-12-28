import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import timm
from sklearn.metrics import r2_score
import wandb
from lejepa_dataset import HybridDEXADataset

# --- Configuration ---
# PATHS
FULL_BODY_MANIFEST = '/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl'
CROPS_MANIFEST = '/net/mraid20/export/genie/LabData/Analyses/gilsa/crops_manifest.pkl'
TARGETS_CSV = '/net/mraid20/export/genie/LabData/Analyses/gilsa/csv_files/targets_for_dino.csv'

# STATS
DEXA_MEAN = [0.12394659966230392, 0.2626885771751404, 0.3075794577598572]
DEXA_STD  = [0.21822425723075867, 0.31778785586357117, 0.3350508213043213]

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-3
PROBE_LR = 1e-3
WD = 1e-3
EPOCHS = 400
LAMBDA = 0.8
SIGREG_SLICES = 2048
WARMUP_EPOCHS = 20
RESUME_FROM = None
MODEL = 'resnet50'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Data Preparation Logic ---

def prepare_combined_manifest():
    print("Loading Manifests...")
    fb_df = pd.read_pickle(FULL_BODY_MANIFEST)

    # Ensure ID formatting matches
    fb_df['RegistrationCode'] = fb_df['RegistrationCode'].astype(str).apply(lambda x: f"10K_{x}" if not x.startswith("10K_") else x)
    fb_df['research_stage']  = fb_df['research_stage'].replace('00_00_visit', 'baseline')

    # Load Crops
    crops_df = pd.read_pickle(CROPS_MANIFEST)
    crops_df['RegistrationCode'] = crops_df['RegistrationCode'].astype(str).apply(lambda x: f"10K_{x}" if not x.startswith("10K_") else x)

    # Group Crops by Visit
    crops_grouped = crops_df.groupby(['RegistrationCode', 'research_stage'])['FilePath'].apply(list).to_dict()

    # Add crops column to full body df
    def get_crops(row):
        key = (row['RegistrationCode'], row['research_stage'])
        return crops_grouped.get(key, [])

    fb_df['Crop_Paths'] = fb_df.apply(get_crops, axis=1)
    return fb_df

# --- 2. Transforms & Dataset ---

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
            transforms.RandomResizedCrop(global_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
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
            transforms.RandomResizedCrop(local_size, scale=(0.15, 0.35), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])

# --- 3. Models & Loss ---

def sigreg_loss(z, num_slices=2048, integration_limit=5, n_points=17):
    B, D = z.shape
    device = z.device
    A = torch.randn(D, num_slices, device=device)
    A = A / (A.norm(dim=0, keepdim=True) + 1e-6)
    z_proj = z @ A
    t = torch.linspace(-integration_limit, integration_limit, n_points, device=device).view(1, 1, -1)
    val = t * z_proj.unsqueeze(-1)
    ecf = torch.exp(1j * val).mean(dim=0)
    target_cf = torch.exp(-0.5 * t.squeeze()**2).view(1, -1)
    diff = (ecf - target_cf).abs().pow(2)
    weighted_diff = diff * target_cf
    loss = torch.trapz(weighted_diff, t.squeeze(), dim=-1).mean()
    return loss

class LeJEPA_Encoder(nn.Module):
    def __init__(self, model_name='resnet50', proj_out_dim=256):
        super().__init__()
        is_vit = 'vit' in model_name or 'swin' in model_name
        model_kwargs = {"pretrained": False, "num_classes": 0}
        if is_vit: model_kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(model_name, **model_kwargs)
        self.embed_dim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, proj_out_dim , bias=False),
            nn.BatchNorm1d(proj_out_dim)
        )
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections

class OnlineLinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)
    def forward(self, x): return self.probe(x)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if len(param.shape) == 1 or name.endswith(".bias"): no_decay.append(param)
        else: decay.append(param)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]

def lejepa_collate_fn(batch):
    views_lists, targets = zip(*batch)
    collated_targets = torch.stack(targets)
    num_views = len(views_lists[0])
    collated_views = []
    for i in range(num_views):
        view_batch = [patient_views[i] for patient_views in views_lists]
        collated_views.append(torch.stack(view_batch))
    return collated_views, collated_targets

class ValDEXADataset(Dataset):
    def __init__(self, manifest_df, targets_df):
        self.manifest = manifest_df
        self.targets = targets_df
        # Deterministic transform for Probe validation
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD)
        ])

    def __len__(self):
        return len(self.manifest)

    def _normalize_to_uint8(self, arr):
        arr = arr.astype(np.float32)
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-6: arr = (arr - min_val) / (max_val - min_val)
        else: arr = np.zeros_like(arr)
        return (arr * 255).astype(np.uint8)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        reg_code = row['RegistrationCode']
        visit = row['research_stage']
        target_val = self.targets.loc[(reg_code, visit), 'age']
        if isinstance(target_val, pd.Series): target_val = target_val.iloc[0]
        target_tensor = torch.tensor(target_val, dtype=torch.float32)

        bone_arr   = np.load(row['Path_Bone'])
        tissue_arr = np.load(row['Path_Tissue'])
        comp_arr   = np.load(row['Path_Composite'])

        bone_img   = self._normalize_to_uint8(bone_arr)
        tissue_img = self._normalize_to_uint8(tissue_arr)
        comp_img   = self._normalize_to_uint8(comp_arr)

        h, w = comp_img.shape[:2]
        def match_size(img, th, tw):
            if img.shape[0] != th or img.shape[1] != tw:
                return np.array(Image.fromarray(img).resize((tw, th), Image.BICUBIC))
            return img

        bone_img   = match_size(bone_img, h, w)
        tissue_img = match_size(tissue_img, h, w)

        fb_array = np.stack([bone_img[..., 0], tissue_img[..., 0], comp_img[..., 0]], axis=0)
        fb_tensor = torch.tensor(fb_array, dtype=torch.uint8)
        final_img = self.transform(fb_tensor)
        return [final_img], target_tensor

# --- 5. Helper: Offline Evaluation ---
def run_offline_sanity_check(encoder, train_loader, val_loader, device):
    print("\n[Offline Eval] Starting Ridge Regression Sanity Check...")
    encoder.eval()

    train_feats = []
    train_targets = []

    with torch.no_grad():
        for views, targets in train_loader:
            img = views[0].to(device)
            feats, _ = encoder(img)
            train_feats.append(feats.cpu().numpy())
            train_targets.append(targets.numpy())

    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_targets)

    val_feats = []
    val_targets = []

    with torch.no_grad():
        for views, targets in val_loader:
            img = views[0].to(device)
            feats, _ = encoder(img)
            val_feats.append(feats.cpu().numpy())
            val_targets.append(targets.numpy())

    X_val = np.vstack(val_feats)
    y_val = np.concatenate(val_targets)

    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)

    train_r2 = clf.score(X_train, y_train)
    val_r2 = clf.score(X_val, y_val)

    print(f"[Offline Eval] Ridge Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f}")

    wandb.log({
        "offline_ridge_train_r2": train_r2,
        "offline_ridge_val_r2": val_r2
    })
    return val_r2

# --- 6. Main ---
def main():
    wandb.init(entity="gilsasson12-weizmann-institute-of-science", project="LeJEPA_DEXA_Scratch", name="LeJEPA_RealCrops_ValCheck", resume="allow")

    # --- Data Loading ---
    manifest = prepare_combined_manifest()
    targets = pd.read_csv(TARGETS_CSV, index_col=[0, 1])

    if targets.index.names != ['RegistrationCode', 'research_stage']:
        targets = targets.set_index(['RegistrationCode', 'research_stage'])
    targets.sort_index(inplace=True)
    valid_target_keys = set(targets.index)

    mask = manifest.apply(lambda r: (r['RegistrationCode'], r['research_stage']) in valid_target_keys, axis=1)
    manifest = manifest[mask].reset_index(drop=True)

    # 1. Split Subjects FIRST
    unique_subjects = manifest['RegistrationCode'].unique()
    train_ids, val_ids = train_test_split(unique_subjects, test_size=0.2, random_state=42)

    # 2. Separate Manifests
    train_manifest = manifest[manifest['RegistrationCode'].isin(train_ids)]
    val_manifest = manifest[manifest['RegistrationCode'].isin(val_ids)]

    # 3. Calculate Age Stats ONLY on Training Data
    # Identify which target rows belong to the training subjects
    train_indices = targets.index.get_level_values(0).isin(train_ids)
    train_target_subset = targets.loc[train_indices]

    age_mean = train_target_subset['age'].mean()
    age_std = train_target_subset['age'].std()

    print(f"Age Stats (Train Only) -> Mean: {age_mean:.4f}, Std: {age_std:.4f}")

    # 4. Normalize All Targets using Train Stats
    # (We operate on the original dataframe so both train/val lookups work)
    targets['age'] = (targets['age'] - age_mean) / age_std

    # --- Dataset Creation ---
    train_config = LeJEPATransformConfig()

    # Train Dataset
    train_dataset = HybridDEXADataset(train_manifest, targets, train_config, n_global=2, n_local=8)

    # Val Dataset 1: For SSL Loss (Generates Views like training)
    val_ssl_dataset = HybridDEXADataset(val_manifest, targets, train_config, n_global=2, n_local=8)

    # Val Dataset 2: For Linear Probe (Generates 1 clean image)
    val_probe_dataset = ValDEXADataset(val_manifest, targets)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, collate_fn=lejepa_collate_fn)

    # We use lejepa_collate_fn for ssl validation as well
    val_ssl_loader = DataLoader(val_ssl_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, collate_fn=lejepa_collate_fn)

    val_probe_loader = DataLoader(val_probe_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lejepa_collate_fn)

    # --- Setup Model ---
    encoder = LeJEPA_Encoder(MODEL, proj_out_dim=128).to(DEVICE)
    probe = OnlineLinearProbe(encoder.embed_dim).to(DEVICE)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    opt_enc = optim.AdamW(get_param_groups(encoder, WD), lr=LR)
    opt_probe = optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=1e-3)
    scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)
    mse_crit = nn.MSELoss()

    # Resume
    start_epoch = 0
    best_val_r2 = -float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        probe.load_state_dict(checkpoint['probe'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        print(f"Resumed from Epoch {start_epoch}, Best R2 so far: {best_val_r2:.4f}")

    print("Starting Training...")

    for epoch in range(start_epoch, EPOCHS):
        encoder.train()
        probe.train()
        train_stats = {'loss': 0, 'pred': 0, 'sig': 0, 'probe': 0}
        train_preds, train_trues = [], []

        # --- TRAINING LOOP ---
        for i, (views, age_targets) in enumerate(train_loader):
            n_globals, n_locals = 2, 8
            global_inputs = torch.cat(views[:n_globals], dim=0).to(DEVICE)
            g_feats, g_projs = encoder(global_inputs)

            local_inputs = torch.cat(views[n_globals:], dim=0).to(DEVICE)
            _, l_projs = encoder(local_inputs)

            bs = age_targets.size(0)
            g_projs = g_projs.view(n_globals, bs, -1)
            l_projs = l_projs.view(n_locals, bs, -1)
            view_projections = torch.cat([g_projs, l_projs], dim=0)

            # --- LeJEPA Loss ---
            loss_pred = 0
            n_views_total = n_globals + n_locals
            for g_idx in range(n_globals):
                target_z = view_projections[g_idx]
                for v_idx in range(n_views_total):
                    if v_idx == g_idx: continue
                    loss_pred += mse_crit(view_projections[v_idx], target_z)

            loss_pred /= (n_globals * (n_views_total - 1))

            all_projs_flat = view_projections.reshape(-1, view_projections.shape[-1])
            loss_sigreg = sigreg_loss(all_projs_flat, num_slices=SIGREG_SLICES)

            loss_total = (1 - LAMBDA) * loss_pred + LAMBDA * loss_sigreg

            opt_enc.zero_grad()
            loss_total.backward()
            opt_enc.step()

            # --- Probe Loss ---
            with torch.no_grad():
                feat_for_probe = g_feats[:bs].detach()

            age_pred = probe(feat_for_probe).squeeze()
            age_targets = age_targets.to(DEVICE).view(-1)
            loss_probe = mse_crit(age_pred, age_targets)

            opt_probe.zero_grad()
            loss_probe.backward()
            opt_probe.step()
            scheduler.step()

            train_stats['loss'] += loss_total.item()
            train_stats['pred'] += loss_pred.item()
            train_stats['sig'] += loss_sigreg.item()
            train_stats['probe'] += loss_probe.item()
            train_preds.extend(age_pred.detach().cpu().numpy())
            train_trues.extend(age_targets.cpu().numpy())

        train_r2 = r2_score(train_trues, train_preds)

        # --- VALIDATION (SSL LOSS) ---
        encoder.eval()
        val_ssl_stats = {'loss': 0, 'pred': 0, 'sig': 0}

        with torch.no_grad():
            for views, _ in val_ssl_loader:
                n_globals, n_locals = 2, 8

                # FIX: Process Globals (224x224) and Locals (96x96) separately
                global_inputs = torch.cat(views[:n_globals], dim=0).to(DEVICE)
                _, g_projs = encoder(global_inputs)

                local_inputs = torch.cat(views[n_globals:], dim=0).to(DEVICE)
                _, l_projs = encoder(local_inputs)

                # Now we can combine the PROJECTIONS (vectors), which are the same size
                bs = views[0].size(0)
                g_projs = g_projs.view(n_globals, bs, -1)
                l_projs = l_projs.view(n_locals, bs, -1)
                view_projections = torch.cat([g_projs, l_projs], dim=0)

                # Calc Loss (Same as training)
                loss_pred_val = 0
                n_views_total = n_globals + n_locals
                for g_idx in range(n_globals):
                    target_z = view_projections[g_idx]
                    for v_idx in range(n_views_total):
                        if v_idx == g_idx: continue
                        loss_pred_val += mse_crit(view_projections[v_idx], target_z)

                loss_pred_val /= (n_globals * (n_views_total - 1))

                all_projs_flat_val = view_projections.reshape(-1, view_projections.shape[-1])
                loss_sigreg_val = sigreg_loss(all_projs_flat_val, num_slices=SIGREG_SLICES)
                loss_total_val = (1 - LAMBDA) * loss_pred_val + LAMBDA * loss_sigreg_val

                val_ssl_stats['loss'] += loss_total_val.item()
                val_ssl_stats['pred'] += loss_pred_val.item()
                val_ssl_stats['sig'] += loss_sigreg_val.item()

        avg_val_ssl_loss = val_ssl_stats['loss'] / len(val_ssl_loader)

        # --- VALIDATION (PROBE) ---
        # Checks the downstream task performance
        probe.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for views, age_targets in val_probe_loader:
                img = views[0].to(DEVICE)
                age_targets = age_targets.to(DEVICE).view(-1)
                feats, _ = encoder(img)
                pred = probe(feats).squeeze()
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(age_targets.cpu().numpy())

        val_r2 = r2_score(val_trues, val_preds)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Ep {epoch+1} | Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f} | Train Loss: {train_stats['loss']/len(train_loader):.4f} | Val SSL Loss: {avg_val_ssl_loss:.4f}")

        wandb.log({
            "train_loss": train_stats['loss']/len(train_loader),
            "val_ssl_loss": avg_val_ssl_loss, # <--- NEW: Monitor generalization of pretraining
            "loss_pred": train_stats['pred']/len(train_loader),
            "loss_sig": train_stats['sig']/len(train_loader),
            "train_probe_r2": train_r2,
            "val_probe_r2": val_r2,
            "lr": current_lr
        })

        if val_r2 > best_val_r2:
            print(f"--> New Best Model! Val R2 improved from {best_val_r2:.4f} to {val_r2:.4f}. Saving...")
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'probe': probe.state_dict(),
                'best_val_r2': best_val_r2
            }, "lejepa_best_metric.pth")

        if (epoch + 1) % 100 == 0:
            run_offline_sanity_check(encoder, train_loader, val_probe_loader, DEVICE)
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'probe': probe.state_dict()
            }, f"lejepa_snapshot_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()