import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure you import your updated dataset class
from DEXA.dexa_fm.lejepa_dataset import LeJEPAHDF5Dataset 
from model import LeJEPA_Encoder
from DEXA.dexa_fm.augmentations import val_transforms

# --- CONFIGURATION ---
MODEL_NAME = 'vit_base_patch16_384'
MODEL_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/vit_base_2_gpus_2.pth"
OUTPUT_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/ukbb_embeddings_vit_base.pkl"
# HDF5_PATH = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
HDF5_PATH = '/net/mraid20/export/jasmine/UKBB_DATA/ukbb_dexa_dataset_v3.h5'

BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GLOBAL_VIEWS = 2

class InferenceDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        views, _ = self.base[idx]
        
        self.base._open_h5()
        key = self.base.keys[idx]
        group = self.base.h5_file[key]

        reg_code = group.attrs.get('RegistrationCode', 'Unknown')
        visit_id = group.attrs.get('research_stage', 'Unknown')
        # --- CHANGE 1: Grab Date from HDF5 ---
        date_val = group.attrs.get('Date', 'Unknown')

        if isinstance(reg_code, bytes): reg_code = reg_code.decode('utf-8')
        if isinstance(visit_id, bytes): visit_id = visit_id.decode('utf-8')
        if isinstance(date_val, bytes): date_val = date_val.decode('utf-8')

        # --- CHANGE 2: Return Date ---
        return views, reg_code, visit_id, date_val

def inference_collate_fn(batch):
    # --- CHANGE 3: Unpack Date ---
    views_list, reg_codes, visit_ids, dates = zip(*batch)
    
    patient_stacks = [torch.stack(v) for v in views_list]
    batch_tensor = torch.stack(patient_stacks)
    
    return batch_tensor, reg_codes, visit_ids, dates

def clean_state_dict(checkpoint):
    if 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if "projector" in name or "head" in name:
            continue
        new_state_dict[name] = v
    return new_state_dict

def main():
    print(f"--- Extraction Started on {DEVICE} ---")

    val_transform = val_transforms(global_size=(384, 128))

    base_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=None,
        targets_df=None,
        transform=val_transform,
        n_global=N_GLOBAL_VIEWS,
        n_local=0
    )

    dataset = InferenceDatasetWrapper(base_dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=inference_collate_fn
    )

    print(f"Initializing {MODEL_NAME}...")
    model = LeJEPA_Encoder(MODEL_NAME, img_size=(384, 128), pretrained=False).to(DEVICE)
    model.eval()

    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        clean_dict = clean_state_dict(checkpoint)
        model.load_state_dict(clean_dict, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")

    embeddings_list = []
    indices_list = []

    print("Extracting features...")
    with torch.no_grad():
        for img_stack, reg_codes, visit_ids, dates in tqdm(dataloader):
            
            bs, n_views, c, h, w = img_stack.shape
            flat_imgs = img_stack.view(-1, c, h, w).to(DEVICE)

            output = model(flat_imgs)
            if isinstance(output, tuple): feats = output[0]
            else: feats = output
            
            if feats.dim() == 3: feats = feats[:, 0]

            dim = feats.shape[-1]
            feats = feats.view(bs, n_views, dim)

            # mean pooling across views (if multiple), else just take the single view
            feats = feats.mean(dim=1)

            embeddings_list.append(feats.cpu().numpy())

            # --- CHANGE 5: Zip dates into the index list ---
            for r, v, d in zip(reg_codes, visit_ids, dates):
                indices_list.append((r, v, d))

    print("Compiling DataFrame...")
    all_embeddings = np.vstack(embeddings_list)
    
    # --- CHANGE 6: Add Date to MultiIndex ---
    index = pd.MultiIndex.from_tuples(indices_list, names=['RegistrationCode', 'research_stage', 'Date'])

    # --- BUG FIX: Correct column naming for Pooling ---
    emb_dim = all_embeddings.shape[1] 
    cols = [f"emb_{i}" for i in range(emb_dim)]

    df_emb = pd.DataFrame(all_embeddings, index=index, columns=cols)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_emb.to_pickle(OUTPUT_PATH)

    print(f" DONE. Saved shape {df_emb.shape} to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()