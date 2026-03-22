import os
import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


UKBB_MEAN = [0.13504499197006226, 0.24287331104278564, 0.1884828805923462]
UKBB_STD  = [0.23150044679641724, 0.3116486966609955, 0.2474510669708252]

# --- CONFIGURATION ---
HDF5_PATH = '/net/mraid20/export/jasmine/UKBB_DATA/ukbb_dexa_dataset_v3.h5'
OUTPUT_PATH = '/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/ukbb_embeddings_vit_base.pkl'  # Output file for embeddings (Pickle format)
MODEL_PATH = '/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/vit_base_2_gpus_2.pth'  # Path to the trained LeJEPA model checkpoint
BACKBONE = 'vit_base_patch16_384'  # Example: 'resnet50', 'vit_base_patch16_384'


# Model Settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
IMG_SIZE = (384, 128)

# --- 1. DATASET CLASS (LAZY LOADING) ---
class UKBB_Inference_Dataset(IterableDataset):
    def __init__(self, h5_path, transform=None, mode='all'):
        """
        mode: 'all' (FullBody + Crops), 'fullbody' (No Crops), 'crops' (Only Crops)
        """
        self.h5_path = h5_path
        self.transform = transform
        self.mode = mode
        
        print(f"Reading keys from {h5_path}...")
        with h5py.File(h5_path, 'r') as hf:
            all_keys = list(hf.keys())
        self.all_keys = all_keys
        print(f"Found {len(self.all_keys)} subjects. Inference will start immediately.")

    def process_patient(self, hf, key):
        """Yields one or more samples for a single patient key"""
        samples = []
        try:
            parts = key.split('_')
            eid, visit = parts[0], parts[1]
            
            # --- A. Full Body ---
            if self.mode in ['all', 'fullbody']:
                # Read data
                img_data = hf[key]['fullbody'][:] # (3, H, W)
                
                # Preprocess
                tensor = torch.from_numpy(img_data).float() / 255.0
                if self.transform: tensor = self.transform(tensor)
                
                samples.append({
                    'tensor': tensor,
                    'meta': {'eid': eid, 'visit': visit, 'scan_type': 'composite'}
                })

            # --- B. Crops (Lazy Check) ---
            # We only check for 'crops' NOW, inside the worker loop.
            if self.mode in ['all', 'crops']:
                if 'crops' in hf[key]:
                    crop_grp = hf[key]['crops']
                    for crop_idx in crop_grp.keys():
                        # Read Crop
                        crop_data = crop_grp[crop_idx][:] # (H, W)
                        
                        # Expand to 3 Channels (Ch0=Bone, others=0)
                        h, w = crop_data.shape
                        img_expanded = np.zeros((3, h, w), dtype=crop_data.dtype)
                        img_expanded[0] = crop_data
                        
                        # Preprocess
                        tensor = torch.from_numpy(img_expanded).float() / 255.0
                        if self.transform: tensor = self.transform(tensor)
                        
                        samples.append({
                            'tensor': tensor,
                            'meta': {'eid': eid, 'visit': visit, 'scan_type': f"crop_{crop_idx}"}
                        })
                        
        except Exception as e:
            # print(f"Skipping {key}: {e}")
            pass
            
        return samples

    def __iter__(self):
        """
        This is called when the DataLoader starts.
        It splits the work among workers and streams data.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Sharding: Split keys among workers
        if worker_info is None:
            # Single process
            my_keys = self.all_keys
        else:
            # Multiple workers: Slice the list
            # Worker 0 gets [0, 4, 8...], Worker 1 gets [1, 5, 9...]
            my_keys = self.all_keys[worker_info.id :: worker_info.num_workers]

        # 2. Open HDF5 (Once per worker)
        # We must open a new handle inside the worker process!
        with h5py.File(self.h5_path, 'r') as hf:
            for key in my_keys:
                # Yield all items (FullBody + Crops) for this patient
                items = self.process_patient(hf, key)
                for item in items:
                    yield item['tensor'], item['meta']

# --- 2. MODEL LOADER ---
def load_leJEPA(path, device):
    from model import LeJEPA_Encoder
    print(f"Loading LeJEPA from {path}...")
    
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('encoder', checkpoint)
    
    clean_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        if "projector" not in name and "head" not in name:
            clean_dict[name] = v

    model = LeJEPA_Encoder(model_name=BACKBONE, img_size=IMG_SIZE, pretrained=False)
    msg = model.load_state_dict(clean_dict, strict=False)
    print(f"Model loaded. Missing keys: {len(msg.missing_keys)}")
    
    model.to(device)
    model.eval()
    return model

# --- 3. MAIN EXTRACTION LOOP ---
def main():
    
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=UKBB_MEAN, std=UKBB_STD)
    ])

    # 1. Init Dataset
    dataset = UKBB_Inference_Dataset(HDF5_PATH, transform=tfm, mode='fullbody')  # Options: 'all', 'fullbody', 'crops'
    
    # 2. Init DataLoader
    # Note: No shuffle=True for IterableDataset
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    # 3. Model
    model = load_leJEPA(MODEL_PATH, DEVICE)

    # 4. Inference
    results = []
    print("Starting Extraction...")
    
    # Estimate total batches for tqdm (assuming ~1.5 samples per patient on average due to crops)
    estimated_total = int((len(dataset.all_keys) * 1.5) // BATCH_SIZE)

    with torch.no_grad():
        for imgs, metas in tqdm(loader, total=estimated_total):
            imgs = imgs.to(DEVICE, non_blocking=True)
            
            embeddings = model(imgs)[0].cpu().numpy()
            
            # Unpack metadata (metas is a dict of lists)
            batch_len = len(embeddings)
            for i in range(batch_len):
                results.append({
                    'eid': metas['eid'][i],
                    'visit': metas['visit'][i],
                    'scan_type': metas['scan_type'][i],
                    'embedding': embeddings[i]
                })

    # 5. Save
    if results:
        df = pd.DataFrame(results)
        
        # Restructure: eid/visit as indices, embedding dims as columns
        embeddings_list = df['embedding'].tolist()
        embedding_dim = embeddings_list[0].shape[0]
        embedding_df = pd.DataFrame(embeddings_list, columns=[f'dim_{i}' for i in range(embedding_dim)])
        
        # Add metadata columns
        embedding_df['eid'] = df['eid'].values
        embedding_df['visit'] = df['visit'].values
        embedding_df['scan_type'] = df['scan_type'].values
        
        # Set index based on mode
        if dataset.mode == 'fullbody':
            # Only eid and visit as indices
            embedding_df = embedding_df.set_index(['eid', 'visit'])
            embedding_df = embedding_df.drop(columns=['scan_type'])
        else:  # mode == 'all' or 'crops'
            # Add scan_type as an additional index level
            embedding_df = embedding_df.set_index(['eid', 'visit', 'scan_type'])
        
        print(f"Saving {len(embedding_df)} embeddings to {OUTPUT_PATH}...")
        embedding_df.to_pickle(OUTPUT_PATH)
        print("Success.")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()