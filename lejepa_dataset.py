# import torch
# from torch.utils.data import Dataset
# import h5py
# import numpy as np
# from PIL import Image
# import io
# import pandas as pd

# class LeJEPAHDF5Dataset(Dataset):
#     def __init__(self, hdf5_path, keys=None, targets_df=None, transform=None, n_global=2, n_local=8):
#         self.hdf5_path = hdf5_path
#         self.targets = targets_df
#         self.config = transform
#         self.n_global = n_global
#         self.n_local = n_local
#         if keys is not None:
#             self.keys = keys
#         else:
#             with h5py.File(self.hdf5_path, 'r') as f:
#                 self.keys = list(f.keys())

#         self.h5_file = None

#     def _open_h5(self):
#         if self.h5_file is None:
#             # swmr=True (Single Writer Multiple Reader) makes it robust
#             self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         self._open_h5()
#         key = self.keys[idx]
#         group = self.h5_file[key]

#         # Retrieve Metadata (Stored in Attributes)
#         reg_code = group.attrs['RegistrationCode']
#         visit_id = group.attrs['research_stage']

#         #  Get Target (Label)
#         if self.targets is not None:
#             try:
#                 # Assuming targets_df is indexed by (RegistrationCode, VisitID)
#                 target_val = self.targets.loc[(reg_code, visit_id), 'age']
#                 if isinstance(target_val, (pd.Series, pd.DataFrame)):
#                     target_val = target_val.iloc[0]
#                 target_tensor = torch.tensor(target_val, dtype=torch.float32)
#             except KeyError:
#                 # Handle missing target gracefully (or raise error)
#                 # print(f"Warning: Target not found for ({reg_code}, {visit_id}). Using default -1.")
#                 target_tensor = torch.tensor(-1.0, dtype=torch.float32)
#         else:
#             target_tensor = torch.tensor(0.0, dtype=torch.float32)

#         # 3. Load Full Body
#         # Data is already (3, H, W) uint8
#         fb_data = group['fullbody'][:]

#         # Convert to (H, W, 3) for PIL/Transforms compatibility
#         fb_hwc = np.transpose(fb_data, (1, 2, 0))
#         fb_pil = Image.fromarray(fb_hwc)

#         # --- APPLY TRANSFORMS ---

#         # Mode A: Simple Validation/Test (Return 1 image)
#         if callable(self.config) and not hasattr(self.config, 'global_trans'):
#             return self.config(fb_pil)

#         views = []

#         # 4. Global Views (From Full Body)
#         for _ in range(self.n_global):
#             views.append(self.config.global_trans(fb_pil))

#         # 5. Local Views (From Crops)
#         real_crops = []
#         if 'crops' in group:
#             crop_grp = group['crops']
#             # Load all available crops
#             for k in crop_grp.keys():
#                 c_arr = crop_grp[k][:] # (H, W) uint8
#                 real_crops.append(c_arr)

#         # Sampling Logic
#         selected_crops = []
#         if len(real_crops) > 0:
#             if len(real_crops) > self.n_local:
#                 # Randomly select N crops
#                 indices = np.random.choice(len(real_crops), self.n_local, replace=False)
#                 selected_crops = [real_crops[i] for i in indices]
#             else:
#                 selected_crops = real_crops

#         # Process Real Crops
#         real_count = 0
#         for c_arr in selected_crops:
#             # c_arr is (H, W) grayscale
#             c_pil = Image.fromarray(c_arr)

#             # Apply Local Transform (Augmentation)
#             # This returns a Tensor [C, H, W]
#             bone_tensor_aug = self.config.local_trans(c_pil)

#             # Create 3-channel placeholder [3, H, W] to match model input
#             c, h, w = bone_tensor_aug.shape
#             combined_tensor = torch.zeros((3, h, w), dtype=torch.float32)

#             # Insert Bone into Channel 0
#             combined_tensor[0] = bone_tensor_aug.squeeze(0)

#             # Normalize
#             final_view = self.config.normalize(combined_tensor)
#             views.append(final_view)
#             real_count += 1

#         # Synthetic Crops (If missing real ones)
#         needed = self.n_local - real_count
#         for _ in range(needed):
#             # Crop from the Full Body image
#             views.append(self.config.synthetic_local_trans(fb_pil))

#         return views, target_tensor


import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import pandas as pd

class LeJEPAHDF5Dataset(Dataset):
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
            self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest')

    def _to_3channel(self, arr_2d):
        """
        Takes a 2D (H, W) array and converts to 3-channel RGB PIL Image
        by replicating the channel (R=G=B).
        """
        img_pil = Image.fromarray(arr_2d)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        return img_pil

    def __len__(self):
        return len(self.keys)
 
    def __getitem__(self, idx):
        self._open_h5()
        key = self.keys[idx]
        group = self.h5_file[key]

        # 1. Retrieve Metadata & Target
        reg_code = group.attrs.get('RegistrationCode', None)
        visit_id = group.attrs.get('research_stage', None)

        target_tensor = torch.tensor(-1.0, dtype=torch.float32)
        if self.targets is not None and reg_code is not None:
            try:
                if (reg_code, visit_id) in self.targets.index:
                    target_val = self.targets.loc[(reg_code, visit_id), 'age']
                    if isinstance(target_val, (pd.Series, pd.DataFrame)):
                        target_val = target_val.iloc[0]
                    target_tensor = torch.tensor(target_val, dtype=torch.float32)
            except Exception:
                pass

        # 2. LOAD SCANS (Directly as 2D arrays, no shapes need to match!)
        bone_arr = group['bone'][:]
        tissue_arr = group['tissue'][:]
        
        # Convert to 3-Channel PIL Images (Replication)
        bone_pil = self._to_3channel(bone_arr)
        tissue_pil = self._to_3channel(tissue_arr)

        # --- APPLY TRANSFORMS ---

        # Mode A: Simple Validation (Return just Bone)
        if callable(self.config) and not hasattr(self.config, 'global_trans'):
            return self.config(bone_pil), target_tensor

        views = []

        # 3. Global Views (Mixed!)
        # View 1 -> Bone
        # View 2 -> Tissue
        
        if self.n_global >= 1:
            views.append(self.config.global_trans(bone_pil))
        if self.n_global >= 2:
            views.append(self.config.global_trans(tissue_pil))
            
        # If n_global > 2, random mix
        for _ in range(self.n_global - 2):
            source = bone_pil if np.random.rand() > 0.5 else tissue_pil
            views.append(self.config.global_trans(source))

        # 4. Local Views (Crops)
        # Note: Your creation code says crops are "local bone scans"
        real_crops = []
        if 'crops' in group:
            crop_grp = group['crops']
            for k in crop_grp.keys():
                c_arr = crop_grp[k][:] # (H, W)
                real_crops.append(c_arr)

        # Sampling Logic
        selected_real_crops = []
        if len(real_crops) > 0:
            if len(real_crops) > self.n_local:
                indices = np.random.choice(len(real_crops), self.n_local, replace=False)
                selected_real_crops = [real_crops[i] for i in indices]
            else:
                selected_real_crops = real_crops

        # Process Real Crops
        real_count = 0
        for c_arr in selected_real_crops:
            c_pil = self._to_3channel(c_arr)
            views.append(self.config.local_trans(c_pil))
            real_count += 1

        # Synthetic Crops (Fill needed spots with crops from Bone/Tissue fullbodies)
        needed = self.n_local - real_count
        for _ in range(needed):
            source = bone_pil if np.random.rand() > 0.5 else tissue_pil
            views.append(self.config.synthetic_local_trans(source))

        return views, target_tensor