import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from model import LeJEPA_Encoder

# STATS
DEXA_MEAN = [0.12394659966230392, 0.2626885771751404, 0.3075794577598572]
DEXA_STD  = [0.21822425723075867, 0.31778785586357117, 0.3350508213043213]

# --- CONFIGURATION ---
MODEL_NAME = 'vit_small_patch16_224'
MODEL_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/sweep_winner.pth"
OUTPUT_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/lejepa/sweep_winner.pkl"
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your manifest dataframe (pickle or csv)
MANIFEST_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl"

class DEXAExtractionDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.manifest = manifest_df
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def _normalize_to_uint8(self, arr):
        """Robust normalization to 0-255 uint8"""
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)

        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-6:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = np.zeros_like(arr)

        return (arr * 255).astype(np.uint8)

    def load_image(self, path):
        """Helper to load .npy or image files safely"""
        if path.endswith('.npy'):
            arr = np.load(path)
            return self._normalize_to_uint8(arr)
        else:
            # Fallback for jpg/png
            return np.array(Image.open(path).convert('RGB'))

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        reg_code = row['RegistrationCode']
        visit_id = row['research_stage'] if 'research_stage' in row else 'baseline'

        # 1. Load Data (Bone, Tissue, Composite)
        bone_img = self.load_image(row['Path_Bone'])
        tissue_img = self.load_image(row['Path_Tissue'])
        comp_img = self.load_image(row['Path_Composite'])

        # 2. Resize to Match Composite (Anchor)
        h, w = comp_img.shape[:2]

        # Simple resize helper using PIL for speed/quality
        def resize_np(img, th, tw):
            if img.shape[0] != th or img.shape[1] != tw:
                return np.array(Image.fromarray(img).resize((tw, th), Image.BICUBIC))
            return img

        bone_img = resize_np(bone_img, h, w)
        tissue_img = resize_np(tissue_img, h, w)

        # 3. Stack Channels (Channels First: 3, H, W)
        # Result is (3, H, W)
        img_array = np.stack([bone_img[..., 0], tissue_img[..., 0], comp_img[..., 0]], axis=0)
        img_tensor = torch.tensor(img_array, dtype=torch.uint8)

        # 4. Apply Transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, reg_code, visit_id

def clean_state_dict(checkpoint):
    """
    Fixes the 'projector' size mismatch by removing projector weights.
    We don't need the projector for embeddings anyway.
    """
    if 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # Clean prefix 'module.' (DDP artifact)
        name = k[7:] if k.startswith('module.') else k

        # 1. REMOVE PROJECTOR (Fixes size mismatch error)
        if "projector" in name or "head" in name:
            continue

        new_state_dict[name] = v

    return new_state_dict

def main():
    print(f"--- Setting up Embedding Extraction on {DEVICE} ---")

    # 1. Initialize Model
    # We don't care about proj_dim here because we won't load those weights
    model = LeJEPA_Encoder(MODEL_NAME).to(DEVICE)
    model.eval()

    # 2. Load Weights (With Cleaning)
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # Clean the dictionary to remove the mismatching projector
        clean_dict = clean_state_dict(checkpoint)

        # Load with strict=False (allows missing projector weights)
        msg = model.load_state_dict(clean_dict, strict=False)
        print(f"Weights Loaded. Missing keys (expected): {msg.missing_keys}")
    else:
        raise FileNotFoundError(f"Model path {MODEL_PATH} not found.")

    # 3. Setup Data
    print("Loading Manifest...")
    # Load your manifest however you prefer (pickle, csv)
    # Ensure it has 'Path_Bone', 'Path_Tissue', 'Path_Composite'
    try:
        manifest = pd.read_pickle(MANIFEST_PATH)
    except:
        manifest = pd.read_csv(MANIFEST_PATH) # Fallback

    print(f"Found {len(manifest)} samples.")

    # Define Transform (Standard Validation Preprocessing)
    extract_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        # Use the mean/std we calculated before
        transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD),
    ])

    dataset = DEXAExtractionDataset(manifest, transform=extract_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Extraction Loop
    embeddings_list = []
    indices_list = []

    print("Starting Extraction...")
    with torch.no_grad():
        for imgs, reg_codes, visit_ids in tqdm(dataloader):
            imgs = imgs.to(DEVICE)

            # Forward Pass
            # LeJEPA usually returns (features, projections). We want features.
            output = model(imgs)

            if isinstance(output, tuple):
                feats = output[0] # Backbone features
            else:
                feats = output

            # Handle ViT [CLS] token if needed
            if feats.dim() == 3:
                # (B, N, D) -> Take CLS token at index 0
                feats = feats[:, 0]

            embeddings_list.append(feats.cpu().numpy())

            # Store IDs
            for r, v in zip(reg_codes, visit_ids):
                indices_list.append((r, v))

    # 5. Save Results
    print("Compiling Results...")
    all_embeddings = np.vstack(embeddings_list)

    # Create MultiIndex
    index = pd.MultiIndex.from_tuples(indices_list, names=['RegistrationCode', 'research_stage'])

    # Create DataFrame
    # Columns: emb_0, emb_1, ...
    cols = [f"emb_{i}" for i in range(all_embeddings.shape[1])]
    df_emb = pd.DataFrame(all_embeddings, index=index, columns=cols)

    # Create directory if missing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df_emb.to_pickle(OUTPUT_PATH)
    print(f"SUCCESS: Saved {len(df_emb)} embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()