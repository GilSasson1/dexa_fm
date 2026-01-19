import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import LeJEPA_Encoder
from augmentations import val_transforms
from lejepa_dataset import LeJEPAHDF5Dataset

# --- CONFIGURATION ---
MODEL_NAME = 'vit_small_patch16_224'
MODEL_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/best_model_new_dataset_test_final.pth"
OUTPUT_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/lejepa/vit_small_new_data.pkl"
HDF5_PATH = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'

BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GLOBAL_VIEWS = 4 # TTA Views

class InferenceDatasetWrapper(Dataset):
    """
    Wraps the standard LeJEPAHDF5Dataset to return Metadata (IDs)
    alongside the images.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # 1. Get the standard data (views, target)
        # We ignore 'target' for extraction
        views, _ = self.base[idx]

        # 2. Get the Metadata manually
        # We access the internal keys/file of the base dataset
        self.base._open_h5()
        key = self.base.keys[idx]
        group = self.base.h5_file[key]

        # Extract IDs
        # Note: Adjust attribute names if your HDF5 uses different casing
        reg_code = group.attrs.get('RegistrationCode', 'Unknown')
        visit_id = group.attrs.get('research_stage', 'Unknown')

        # Convert strings if they are bytes
        if isinstance(reg_code, bytes): reg_code = reg_code.decode('utf-8')
        if isinstance(visit_id, bytes): visit_id = visit_id.decode('utf-8')

        return views, reg_code, visit_id

def inference_collate_fn(batch):
    """
    Custom collate to handle the list of views.
    Returns:
        collated_views: Tensor of shape (Batch_Size, N_Views, 3, H, W)
        reg_codes: List
        visit_ids: List
    """
    views_list, reg_codes, visit_ids = zip(*batch)

    # views_list is a tuple of lists: ( [View1_A, View2_A...], [View1_B, View2_B...] )
    # We want to stack them: (Batch, N_Views, C, H, W)

    # 1. Stack views for each patient -> (N_Views, C, H, W)
    patient_stacks = [torch.stack(v) for v in views_list]

    # 2. Stack patients -> (Batch, N_Views, C, H, W)
    batch_tensor = torch.stack(patient_stacks)

    return batch_tensor, reg_codes, visit_ids

def clean_state_dict(checkpoint):
    """Removes projector weights."""
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

    # 1. Setup Dataset
    val_transform = val_transforms(global_size=(384, 128))

    base_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=None,  # Use all keys
        targets_df=None, # No targets needed for extraction
        transform=val_transform,
        n_global=N_GLOBAL_VIEWS,
        n_local=0
    )

    # Wrap it to get IDs
    dataset = InferenceDatasetWrapper(base_dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=inference_collate_fn
    )

    # 2. Setup Model
    print(f"Initializing {MODEL_NAME}...")
    model = LeJEPA_Encoder(MODEL_NAME, img_size=(384, 128)).to(DEVICE)
    model.eval()

    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        clean_dict = clean_state_dict(checkpoint)
        msg = model.load_state_dict(clean_dict, strict=False)
        print(f"Weights Loaded. Missing keys (expected): {msg.missing_keys}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")

    # 4. Extraction Loop
    embeddings_list = []
    indices_list = []

    print("Extracting features...")
    with torch.no_grad():
        for img_stack, reg_codes, visit_ids in tqdm(dataloader):

            # img_stack shape: (Batch, N_Views, 3, H, W)
            bs, n_views, c, h, w = img_stack.shape

            # Flatten to feed into model: (Batch * N_Views, 3, H, W)
            flat_imgs = img_stack.view(-1, c, h, w).to(DEVICE)

            # Forward Pass
            output = model(flat_imgs)

            # Handle output format
            if isinstance(output, tuple):
                feats = output[0]
            else:
                feats = output

            # Global Average Pooling if needed (Check dimensions)
            # ViT Small usually returns (B, Dim) for CLS if pooled=True in timm,
            # or (B, N_Patches, Dim). LeJEPA_Encoder usually returns (B, Dim) CLS.
            if feats.dim() == 3:
                # Take CLS token (Index 0)
                feats = feats[:, 0]

            # Reshape back to (Batch, N_Views, Dim)
            dim = feats.shape[-1]
            feats = feats.view(bs, n_views, dim)

            # MEAN POOLING over the N_Global Views (TTA)
            # This gives one robust embedding per patient
            feats_avg = feats.mean(dim=1)

            embeddings_list.append(feats_avg.cpu().numpy())

            # Collect Index
            for r, v in zip(reg_codes, visit_ids):
                indices_list.append((r, v))

    #  Save Results
    print("Compiling DataFrame...")
    all_embeddings = np.vstack(embeddings_list)

    # Create Index
    index = pd.MultiIndex.from_tuples(indices_list, names=['RegistrationCode', 'research_stage'])

    # Columns
    cols = [f"emb_{i}" for i in range(all_embeddings.shape[1])]
    df_emb = pd.DataFrame(all_embeddings, index=index, columns=cols)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_emb.to_pickle(OUTPUT_PATH)

    print(f" DONE. Saved {len(df_emb)} embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()