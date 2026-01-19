import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

# --- Configuration ---
FULL_BODY_SCANS_CACHE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl"
IMG_HEIGHT = 416
IMG_WIDTH = 128

class DEXADataset(Dataset):
    def __init__(self, samples, targets_df=None, target_cols=None, transform=None):

        self.samples = samples
        self.targets_df = targets_df
        self.target_cols = target_cols # Store the list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        sample_id = row['RegistrationCode']
        visit_id = row['research_stage'] if 'research_stage' in row and row['research_stage'] != '00_00_visit' else 'baseline'

        try:
            # 1. Load Data
            bone_np = np.load(row['Path_Bone']).astype(np.float32) / 255.0
            tissue_np = np.load(row['Path_Tissue']).astype(np.float32) / 255.0
            comp_np = np.load(row['Path_Composite']).astype(np.float32) / 255.0

            if bone_np.ndim < 2: raise ValueError("Corrupt Image")

            # 2. Convert to Tensor [C, H, W]
            t_bone = torch.tensor(bone_np).unsqueeze(0)
            t_tissue = torch.tensor(tissue_np).unsqueeze(0)
            t_comp = torch.tensor(comp_np).unsqueeze(0)

            # This ensures all channels are (416, 128) and aligned before concatenation
            target_size = (IMG_HEIGHT, IMG_WIDTH)

            t_bone = TF.resize(t_bone, target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            t_tissue = TF.resize(t_tissue, target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            t_comp = TF.resize(t_comp, target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

            # 4. Concatenate
            img_tensor = torch.cat([t_bone, t_tissue, t_comp], dim=0)

            if self.transform:
                img_tensor = self.transform(img_tensor)

            if self.targets_df is None or self.target_cols is None:
                return img_tensor, (sample_id, visit_id)
            # 1. Robust Lookup
            try:
                # Try MultiIndex
                labels_row = self.targets_df.loc[(sample_id, visit_id)]
            except KeyError:
                # Try Single Index
                labels_row = self.targets_df.loc[sample_id]

            # 2. Handle Duplicates (DataFrame -> Series)
            if isinstance(labels_row, pd.DataFrame):
                labels_row = labels_row.iloc[0]

            # This extracts [Age, BMI, ...] based on your config list
            # .values converts Series -> Numpy Array -> Float
            target_values = labels_row[self.target_cols].values.astype(np.float32)

            # Return shape: [C, H, W], [Num_Targets]
            return img_tensor, torch.tensor(target_values)

        except Exception as e:
            print(f"Error loading sample {sample_id} at index {idx}: {e}")
            raise e

# Calculate mean and std for normalization

def main():
    print("Loading Manifest...")
    manifest = pd.read_pickle(FULL_BODY_SCANS_CACHE)
    # Ensure IDs match your format
    manifest['RegistrationCode'] = manifest['RegistrationCode'].astype(str).apply(
        lambda x: f"10K_{x}" if not x.startswith("10K_") else x
    )

    # We don't need targets for this, just images
    # We can pass None for targets if your dataset class handles it (or pass a dummy df)

    # Init Dataset (No Transform! We want raw pixel values [0, 1])
    dataset = DEXADataset(
        manifest,
        targets_df=None,
        transform=None  # Important: No Normalize here!
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Calculating stats for {len(dataset)} images...")

    # Placeholders
    # We use the Welford's algorithm or simple sum/sq_sum for stability
    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_pixels = 0

    for i, (imgs, _) in enumerate(tqdm(loader)):
        # imgs shape: [B, 3, H, W]
        # We want to aggregate over [B, H, W] dimensions

        # 1. Sum over batch, height, width
        channels_sum += torch.sum(imgs, dim=[0, 2, 3])

        # 2. Sum of squares
        channels_sq_sum += torch.sum(imgs ** 2, dim=[0, 2, 3])

        # 3. Count total pixels
        # B * H * W
        num_pixels += imgs.size(0) * imgs.size(2) * imgs.size(3)

    # Final Calculation
    mean = channels_sum / num_pixels

    # Std = sqrt( E[x^2] - (E[x])^2 )
    std = (channels_sq_sum / num_pixels - mean ** 2) ** 0.5

    print("\n--- RESULTS ---")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")

    # Format for copy-pasting
    print(f"\nDEXA_MEAN = {mean.tolist()}")
    print(f"DEXA_STD  = {std.tolist()}")

if __name__ == "__main__":
    main()
