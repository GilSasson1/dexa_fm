import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF
import h5py

# --- Configuration ---
HDF5_PATH = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5"
IMG_HEIGHT = 256
IMG_WIDTH = 256

class DEXADataset(Dataset):
    def __init__(self, hdf5_path, keys, targets_df=None, target_cols=None, transform=None):
        self.hdf5_path = hdf5_path
        self.keys = keys
        self.targets_df = targets_df
        self.target_cols = target_cols # Store the list
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
        parts = key.split('_')
        sample_id = "_".join(parts[:2])
        visit_id = "_".join(parts[2:])

        try:
            # 1. Load Data from HDF5
            bone_np = self.h5_file[key]['bone'][:].astype(np.float32) / 255.0
            tissue_np = self.h5_file[key]['tissue'][:].astype(np.float32) / 255.0

            if bone_np.ndim < 2: raise ValueError("Corrupt Image")

            # 2. Convert to Tensor [C, H, W]
            t_bone = torch.tensor(bone_np).unsqueeze(0)
            t_tissue = torch.tensor(tissue_np).unsqueeze(0)
            # t_comp = torch.tensor(comp_np).unsqueeze(0)

            # This ensures all channels are (256, 256) and aligned before concatenation
            target_size = (IMG_HEIGHT, IMG_WIDTH)

            t_bone = TF.resize(t_bone, target_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            t_tissue = TF.resize(t_tissue, target_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            # t_comp = TF.resize(t_comp, target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

            # --- ORIGINAL: 3-channel stacking (commented out) ---
            # # 4. Concatenate
            # img_tensor = torch.cat([t_bone, t_tissue, t_comp], dim=0)

            # --- EXPERIMENT: Separate bone and tissue as 3-channel images ---
            # Replicate bone scan to 3 channels (for bone embedding)
            bone_3ch = torch.cat([t_bone, t_bone, t_bone], dim=0)
            # Replicate tissue scan to 3 channels (for tissue embedding)
            tissue_3ch = torch.cat([t_tissue, t_tissue, t_tissue], dim=0)

            if self.transform:
                bone_3ch = self.transform(bone_3ch)
                tissue_3ch = self.transform(tissue_3ch)

            if self.targets_df is None or self.target_cols is None:
                # Return both image tensors for separate embeddings
                return (bone_3ch, tissue_3ch), (sample_id, visit_id)
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

            # Return shape: (bone_3ch, tissue_3ch), [Num_Targets]
            return (bone_3ch, tissue_3ch), torch.tensor(target_values)

        except Exception as e:
            print(f"Error loading sample {sample_id} at index {idx}: {e}")
            raise e

# Calculate mean and std for normalization

def main():
    print(f"Loading HDF5 Keys from {HDF5_PATH}...")
    with h5py.File(HDF5_PATH, 'r') as f:
        keys = list(f.keys())

    # Init Dataset (No Transform! We want raw pixel values [0, 1])
    dataset = DEXADataset(
        hdf5_path=HDF5_PATH,
        keys=keys,
        targets_df=None,
        transform=None  # Important: No Normalize here!
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Calculating stats for {len(dataset)} images...")

    bone_sum = torch.zeros(3)
    bone_sq_sum = torch.zeros(3)
    tissue_sum = torch.zeros(3)
    tissue_sq_sum = torch.zeros(3)
    num_pixels = 0

    for i, ((bone_imgs, tissue_imgs), _) in enumerate(tqdm(loader)):
        bone_sum += torch.sum(bone_imgs, dim=[0, 2, 3])
        bone_sq_sum += torch.sum(bone_imgs ** 2, dim=[0, 2, 3])

        tissue_sum += torch.sum(tissue_imgs, dim=[0, 2, 3])
        tissue_sq_sum += torch.sum(tissue_imgs ** 2, dim=[0, 2, 3])

        num_pixels += bone_imgs.size(0) * bone_imgs.size(2) * bone_imgs.size(3)

    bone_mean = bone_sum / num_pixels
    bone_std = (bone_sq_sum / num_pixels - bone_mean ** 2) ** 0.5

    tissue_mean = tissue_sum / num_pixels
    tissue_std = (tissue_sq_sum / num_pixels - tissue_mean ** 2) ** 0.5

    print("\n--- RESULTS ---")
    print(f"BONE_MEAN = {bone_mean.tolist()}")
    print(f"BONE_STD  = {bone_std.tolist()}")
    print(f"TISSUE_MEAN = {tissue_mean.tolist()}")
    print(f"TISSUE_STD  = {tissue_std.tolist()}")

if __name__ == "__main__":
    main()
