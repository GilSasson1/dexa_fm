from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from PIL import Image

class LeJEPADEXADataset(Dataset):
    def __init__(self, manifest_df, targets_df, transform_config, n_global=2, n_local=8):
        self.manifest = manifest_df
        self.targets = targets_df
        self.config = transform_config
        self.n_global = n_global
        self.n_local = n_local

        # Pre-calculate valid indices to speed up random sampling if needed
        self.indices = list(range(len(self.manifest)))

    def __len__(self):
        return len(self.manifest)

    def _normalize_to_uint8(self, arr):
        """
        Robustly normalizes any raw float/int numpy array to 0-255 uint8.
        """
        arr = arr.astype(np.float32)

        # Handle 3D (H,W,C) or 2D (H,W)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)

        # Min-Max Normalization
        min_val = arr.min()
        max_val = arr.max()

        # Avoid division by zero if image is completely flat (e.g. all zeros)
        if max_val - min_val > 1e-6:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            # If image is constant, return black (zeros) or handled upstream
            # But technically a constant image is a valid uint8 image
            arr = np.zeros_like(arr)

        # Scale to 0-255
        arr = (arr * 255).astype(np.uint8)
        return arr

    def __getitem__(self, idx):
        current_idx = idx
        row = self.manifest.iloc[current_idx]
        reg_code = row['RegistrationCode']
        visit = row['research_stage']

        target_val = self.targets.loc[(reg_code, visit), 'age']
        if isinstance(target_val, pd.Series):
            target_val = target_val.iloc[0] # Take the first if duplicate
        target_tensor = torch.tensor(target_val, dtype=torch.float32)

        views = []

        #  Load Full Body Image
        bone_path  = row['Path_Bone']
        tissue_path = row['Path_Tissue']
        comp_path = row['Path_Composite']

        bone_arr = np.load(bone_path)
        tissue_arr = np.load(tissue_path)
        comp_arr = np.load(comp_path)

        # Normalize each channel to uint8
        bone_img = self._normalize_to_uint8(bone_arr)
        tissue_img = self._normalize_to_uint8(tissue_arr)
        comp_img = self._normalize_to_uint8(comp_arr)

        h, w = comp_img.shape[:2]

        def resize_if_needed(img_arr, target_h, target_w):
            if img_arr.shape[0] != target_h or img_arr.shape[1] != target_w:
                # Convert to PIL, resize, convert back
                return np.array(Image.fromarray(img_arr).resize((target_w, target_h), Image.BICUBIC))
            return img_arr

        bone_img   = resize_if_needed(bone_img, h, w)
        tissue_img = resize_if_needed(tissue_img, h, w)

        # Stack channels: R=Bone, G=Tissue, B=Composite
        fb_array = np.stack([bone_img[..., 0], tissue_img[..., 0], comp_img[..., 0]], axis=0) # Shape: (3, H, W)
        fb_tensor = torch.tensor(fb_array, dtype=torch.uint8)
        # Pre-process Full Body (Squash)
        fb_squashed = self.config.fb_pre(fb_tensor)

        # Generate Global Views
        for _ in range(self.n_global):
            views.append(self.config.global_trans(fb_squashed))

        # Generate Local Views
        crop_paths = row['Crop_Paths'] # List of paths from manifest
        if len(crop_paths) > self.n_local:
            crop_paths = np.random.choice(crop_paths, size=self.n_local, replace=False)
        real_crops_loaded = []

        # Load available Real Crops
        for crop_path in crop_paths:
            try:
                crop_arr = np.load(crop_path)
                crop_img = self._normalize_to_uint8(crop_arr)
                crop_tensor = torch.tensor(crop_img, dtype=torch.uint8).permute(2, 0, 1) # (C, H, W)
                real_crops_loaded.append(crop_tensor)
            except:
                continue # Skip if loading fails

        # Apply transforms to Real Crops
        for crop_img in real_crops_loaded:
            views.append(self.config.local_trans(crop_img))

        # Fill remainder with Synthetic Crops (from Full Body)
        needed = self.n_local - len(real_crops_loaded)
        for _ in range(needed):
            views.append(self.config.synthetic_local_trans(fb_squashed))

        return views, target_tensor