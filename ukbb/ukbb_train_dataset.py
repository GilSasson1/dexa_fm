import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import pandas as pd

class UKBB_LeJEPADataset(Dataset):
    def __init__(self, hdf5_path, keys=None, targets_df=None, transform=None, n_global=2, n_local=8):
        self.hdf5_path = hdf5_path
        self.targets = targets_df
        self.config = transform
        self.n_global = n_global
        self.n_local = n_local

        if keys is not None:
            self.keys = keys
        else:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.keys = list(f.keys())

        self.h5_file = None

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._open_h5()
        key = self.keys[idx]
        group = self.h5_file[key]

        # 1. Parse Key (e.g. "1234567_2_0")
        try:
            parts = key.split('_')
            eid_str = parts[0]
            # visit_id = "_".join(parts[1:]) # Not needed for single-index lookup
        except Exception:
            eid_str = key

        # Get Label
        if self.targets is not None:
            try:
                # Try looking up by String EID
                target_val = self.targets.loc[eid_str, 'age']
            except KeyError:
                # Try looking up by Integer EID (common mismatch)
                try:
                    target_val = self.targets.loc[int(eid_str), 'age']
                except (KeyError, ValueError):
                    print(f"Warning: Target for EID {eid_str} not found in targets DataFrame.")
                    # Target missing -> Return -1.0
                    target_val = -1.0

            target_tensor = torch.tensor(target_val, dtype=torch.float32)
        else:
            print (f"Warning: No targets DataFrame provided. Returning dummy target for key {key}.")
            target_tensor = torch.tensor(-1.0, dtype=torch.float32)

        # Load Full Body
        # Format is (3, H, W)
        fb_data = group['fullbody'][:]

        # Transpose (3, H, W) -> (H, W, 3) for PIL
        fb_hwc = np.transpose(fb_data, (1, 2, 0))
        fb_pil = Image.fromarray(fb_hwc)


        # Validation Mode
        if callable(self.config) and not hasattr(self.config, 'global_trans'):
            return self.config(fb_pil), target_tensor

        views = []

        # Global Views
        for _ in range(self.n_global):
            views.append(self.config.global_trans(fb_pil))

        # Local Views
        if self.n_local > 0:
            real_crops = []
            if 'crops' in group:
                crop_grp = group['crops']
                for k_crop in crop_grp.keys():
                    c_arr = crop_grp[k_crop][:]
                    real_crops.append(c_arr)

            # Sampling
            selected_crops = []
            if len(real_crops) > 0:
                if len(real_crops) > self.n_local:
                    indices = np.random.choice(len(real_crops), self.n_local, replace=False)
                    selected_crops = [real_crops[i] for i in indices]
                else:
                    selected_crops = real_crops

            real_count = 0
            for c_arr in selected_crops:
                c_pil = Image.fromarray(c_arr) # Mode 'L' (Grayscale)
                bone_tensor_aug = self.config.local_trans(c_pil) # (1, 96, 96)

                # Create 3-channel container
                c, h, w = bone_tensor_aug.shape
                combined_tensor = torch.zeros((3, h, w), dtype=torch.float32)
                combined_tensor[0] = bone_tensor_aug.squeeze(0)

                final_view = self.config.normalize(combined_tensor)
                views.append(final_view)
                real_count += 1

            # Fallback: Synthetic crops
            needed = self.n_local - real_count
            for _ in range(needed):
                views.append(self.config.synthetic_local_trans(fb_pil))

        return views, target_tensor