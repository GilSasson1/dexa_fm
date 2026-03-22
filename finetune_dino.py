import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import h5py
import wandb
from torchvision.transforms import v2
from peft import get_peft_model, LoraConfig
from dinov3_model import DINOv3

# --- CONFIGURATION ---
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FREEZE_BACKBONE = False  # Set to True for Linear Probing, False for Fine-Tuning/PEFT
USE_PEFT = True
FUSION_STRATEGY = 'late_fusion' # Options: 'concat', 'mean_pool', 'late_fusion'
# Paths
HDF5_PATH = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
TARGETS_CSV = '/home/gilsa/PycharmProjects/DEXA/dexa_fm/csvs/targets_for_downstream.csv'
SAVED_MODEL_PATH = '/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/dinov3_full_ft.pth'
FINAL_EMBEDDINGS_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dexa_embeddings_dinov3.pkl"

WANDB_PROJECT = "DEXA_DINO_FineTuning"
WANDB_ENTITY = "gilsasson12-weizmann-institute-of-science"
TARGET_COLUMN = 'age'
RUN_NAME = 'DINOv3_FT_Age_LateFusion_PEFT'  # Example: 'DINOv3_FT_Age_LateFusion_PEFT'
MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'

# ---------------------------------------------------------
#  Transforms (V2)
# ---------------------------------------------------------
def make_transform(resize_size: int = 256):

    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([resize, to_float, normalize])

# ---------------------------------------------------------
#  Dataset
# ---------------------------------------------------------
class DINO_HDF5_Dataset(Dataset):
    def __init__(self, hdf5_path, keys, targets_df=None, target_col=None, transform=None):
        self.hdf5_path = hdf5_path
        self.keys = keys
        self.targets_df = targets_df
        self.target_col = target_col
        self.transform = transform
        self.h5_file = None

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._open_h5()
        key = self.keys[idx]

        # Load Bone and Tissue
        bone_np = self.h5_file[key]['bone'][:]
        tissue_np = self.h5_file[key]['tissue'][:]

        # Convert to Tensor, Add Channel Dim, Scale [0, 255] -> [0.0, 1.0]
        t_bone = torch.from_numpy(bone_np).float().unsqueeze(0) / 255.0
        t_tissue = torch.from_numpy(tissue_np).float().unsqueeze(0) / 255.0
        # Replicate to 3 channels for DINOv3
        bone_3ch = t_bone.repeat(3, 1, 1)
        tissue_3ch = t_tissue.repeat(3, 1, 1)

        #  Apply Transforms
        if self.transform:
            bone_3ch = self.transform(bone_3ch)
            tissue_3ch = self.transform(tissue_3ch)

        label = 0.0
        if self.targets_df is not None:
            parts = key.split('_')
            s_id = "_".join(parts[:2])
            v_id = "_".join(parts[2:])
            val = self.targets_df.loc[(s_id, v_id), self.target_col]
            if isinstance(val, pd.Series): val = val.iloc[0]
            label = float(val)

        return (bone_3ch, tissue_3ch), torch.tensor(label, dtype=torch.float32)

class DINOv3Regressor(nn.Module):
    def __init__(self, base_model, fusion_strategy=FUSION_STRATEGY):
        super().__init__()
        self.backbone = base_model
        self.fusion_strategy = fusion_strategy
        self.embed_dim = base_model.backbone.num_features

        if self.fusion_strategy == 'concat':
            self.head = nn.Linear(self.embed_dim * 2, 1)
        elif self.fusion_strategy == 'mean_pool':
            self.head = nn.Linear(self.embed_dim, 1)
        elif self.fusion_strategy == 'late_fusion':
            self.head_bone = nn.Linear(self.embed_dim, 1)
            self.head_tissue = nn.Linear(self.embed_dim, 1)

    def get_head_params(self):
        if self.fusion_strategy == 'late_fusion':
            return list(self.head_bone.parameters()) + list(self.head_tissue.parameters())
        return list(self.head.parameters())

    def forward(self, bone, tissue):
        feats_bone = self.backbone(bone)
        feats_tissue = self.backbone(tissue)

        if self.fusion_strategy == 'concat':
            combined = torch.cat([feats_bone, feats_tissue], dim=1)
            return self.head(combined)
        elif self.fusion_strategy == 'mean_pool':
            combined = (feats_bone + feats_tissue) / 2.0
            return self.head(combined)
        elif self.fusion_strategy == 'late_fusion':
            pred_bone = self.head_bone(feats_bone)
            pred_tissue = self.head_tissue(feats_tissue)
            return (pred_bone + pred_tissue) / 2.0

    def extract(self, bone, tissue):
        feats_bone = self.backbone(bone)
        feats_tissue = self.backbone(tissue)
        if self.fusion_strategy == 'mean_pool':
            return (feats_bone + feats_tissue) / 2.0
        # Return concat for both concat & late_fusion so downstream tasks have both representations
        return torch.cat([feats_bone, feats_tissue], dim=1) 

def get_model():
    #  Instantiate the base model
    base_model = DINOv3()

    #  Wrap it in our Regressor class
    # This ensures forward() calls backbone() -> head()
    model = DINOv3Regressor(base_model)

    if FREEZE_BACKBONE:
        print("[Config] Linear Probing Mode Enabled (Frozen Backbone)")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.get_head_params():
            param.requires_grad = True
    elif USE_PEFT:
        print("[Config] PEFT Mode Enabled")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none",
        )
        # Apply LoRA specifically to the backbone
        model.backbone = get_peft_model(model.backbone, lora_config)
        model.backbone.print_trainable_parameters()
    else:
        print("[Config] Full Fine-Tuning Mode Enabled")
        for param in model.parameters():
            param.requires_grad = True

    return model.to(DEVICE)

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def run_validation(model, loader, criterion):
    model.eval()
    val_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        for (bone, tissue), label in loader:
            bone, tissue, label = bone.to(DEVICE), tissue.to(DEVICE), label.to(DEVICE).float()
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                output = model(bone, tissue).squeeze()
            val_loss += criterion(output, label).item()
            preds.extend(output.cpu().float().numpy())
            truths.extend(label.cpu().numpy())
    return val_loss / len(loader), r2_score(truths, preds)

def extract_embeddings(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    embs, ids = [], []
    with torch.no_grad():
        for i, ((bone, tissue), _) in enumerate(tqdm(loader, desc="Extracting")):
            bone, tissue = bone.to(DEVICE), tissue.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                feats = model.extract(bone, tissue)
            embs.append(feats.cpu().float().numpy())
            start_idx = i * BATCH_SIZE
            batch_keys = dataset.keys[start_idx : start_idx + len(bone)]
            for k in batch_keys:
                parts = k.split('_')
                ids.append(("_".join(parts[:2]), "_".join(parts[2:])))

    index = pd.MultiIndex.from_tuples(ids, names=['RegistrationCode', 'research_stage'])
    return pd.DataFrame(np.vstack(embs), index=index, columns=[f"emb_{i}" for i in range(embs[0].shape[1])])

# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- DATA LOADING ---
    print(f"Scanning HDF5: {HDF5_PATH}")
    with h5py.File(HDF5_PATH, 'r') as f: all_keys = list(f.keys())

    target_df = pd.read_csv(TARGETS_CSV, index_col=[0, 1]).dropna(subset=[TARGET_COLUMN])
    # nomalize target column for better training stability
    target_df[TARGET_COLUMN] = (target_df[TARGET_COLUMN] - target_df[TARGET_COLUMN].mean()) / target_df[TARGET_COLUMN].std()

    target_df.sort_index(inplace=True)

    valid_keys = []
    for k in all_keys:
        parts = k.split('_')
        idx = ("_".join(parts[:2]), "_".join(parts[2:]))
        if idx in target_df.index: valid_keys.append(k)

    train_keys, val_keys = train_test_split(valid_keys, test_size=0.2, random_state=42)

    # --- TRANSFORMS ---
    train_trans = make_transform(resize_size=256)
    val_trans   = make_transform(resize_size=256)

    ds_train = DINO_HDF5_Dataset(HDF5_PATH, train_keys, target_df, TARGET_COLUMN, transform=train_trans)
    ds_val = DINO_HDF5_Dataset(HDF5_PATH, val_keys, target_df, TARGET_COLUMN, transform=val_trans)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # --- SETUP FOR FULL FINE-TUNING ---
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=RUN_NAME, config={"peft": USE_PEFT, "freeze_backbone": FREEZE_BACKBONE, "fusion_strategy": FUSION_STRATEGY})

    model = get_model()

    if FREEZE_BACKBONE:
        optimizer = optim.AdamW(model.get_head_params(), lr=1e-4, weight_decay=0.01)
    else:
        # Differential Learning Rates
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-6}, # Very small LR for backbone
            {'params': model.get_head_params(),     'lr': 1e-4}  # Larger LR for head
        ], weight_decay=0.01)

    criterion = nn.MSELoss()
    best_r2 = -float('inf')

    print("Starting Full Fine-Tuning...")

    for epoch in range(50):
        model.train()
        train_loss = 0

        for i, ((bone, tissue), label) in enumerate(loader_train):
            bone, tissue, label = bone.to(DEVICE), tissue.to(DEVICE), label.to(DEVICE).float()

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                output = model(bone, tissue).squeeze()
                loss = criterion(output, label)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        v_loss, v_r2 = run_validation(model, loader_val, criterion)
        print(f"Ep {epoch+1} | Loss: {train_loss/len(loader_train):.4f} | Val R2: {v_r2:.4f}")
        wandb.log({"train_loss": train_loss/len(loader_train), "val_r2": v_r2})

        if v_r2 > best_r2:
            best_r2 = v_r2
            torch.save(model.state_dict(), SAVED_MODEL_PATH)
            print("  -> Saved Best Model")

    # --- EXTRACTION ---
    print("Extracting full embeddings...")
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    ds_full = DINO_HDF5_Dataset(HDF5_PATH, valid_keys, target_df, TARGET_COLUMN, transform=val_trans)
    emb_df = extract_embeddings(model, ds_full)
    emb_df.to_pickle(FINAL_EMBEDDINGS_PATH)
    wandb.finish()