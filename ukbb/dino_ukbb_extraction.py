import torch
import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dinov3_model import DINOv3

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
HDF5_PATH = '/net/mraid20/export/jasmine/UKBB_DATA/ukbb_dexa_dataset_v2.h5'
OUTPUT_FILE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/ukbb_dexa_embeddings_dino_2.pkl"

# DINO SPECS
HEIGHT = 256
WIDTH = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- DATASET CLASS (HDF5) ---
class UKBB_HDF5_Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.hf = None  # Handle for lazy loading

        # 1. Quick scan of keys (Metadata only)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as hf:
            self.keys = list(hf.keys())

        print(f"Found {len(self.keys)} subjects in HDF5.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Lazy Loading: Open file strictly inside the worker process
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        key = self.keys[idx]

        try:
            # 1. Load Data
            # HDF5 stores as uint8 (3, H, W). We slice [:] to get numpy array.
            data = self.hf[key]['fullbody'][:]

            # 2. Convert to Tensor (Float 0.0 - 1.0)
            img_tensor = torch.from_numpy(data).float() / 255.0

            # 3. Apply Transforms
            if self.transform:
                img_tensor = self.transform(img_tensor)

            # 4. Parse ID and Visit
            # Key Format: "1234567_2_0" (Eid_Visit)
            parts = key.split('_')
            patient_id = parts[0]
            # Reconstruct visit (everything after the first underscore)
            visit_id = "_".join(parts[1:])

            return img_tensor, patient_id, visit_id

        except Exception as e:
            print(f"Error loading {key}: {e}")
            # Return dummy tensor to keep batch logic alive
            dummy = torch.zeros((3, HEIGHT, WIDTH))
            return dummy, "error", "error"

def extract_features():
    # Explicit tuple casting for size to avoid Torch Tensor errors
    img_size = (HEIGHT, WIDTH)

    transform = transforms.Compose([
        transforms.Resize(tuple(img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Initialize HDF5 Dataset
    dataset = UKBB_HDF5_Dataset(HDF5_PATH, transform=transform)

    # num_workers=4 is safe with the lazy loading implementation
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("Initializing DINO backbone...")
    # Loading logic based on your dinov3_model.py
    model = DINOv3(model_name='vit_small_patch16_dinov3.lvd1689m')
    model = model.to(DEVICE)
    model.eval()

    # --- MEMORY OPTIMIZATION ---
    all_embeddings = []
    all_indices = [] # Will hold tuples of (eid, visit_index)

    print("Starting extraction...")
    with torch.no_grad():
        for images, pids, vids in tqdm(dataloader):
            images = images.to(DEVICE, non_blocking=True)

            try:
                # Some DINO wrappers expose .backbone, others are direct calls
                embeddings = model.backbone(images)
            except AttributeError:
                embeddings = model(images)

            # Move to CPU numpy immediately to free GPU memory
            batch_emb = embeddings.cpu().numpy()

            for i, (pid, vid) in enumerate(zip(pids, vids)):
                if pid != "error":
                    all_embeddings.append(batch_emb[i])
                    all_indices.append((pid, vid))

    print(f"Extraction complete. Found {len(all_embeddings)} embeddings.")

    # --- EFFICIENT DATAFRAME CREATION ---
    if len(all_embeddings) > 0:
        print("Stacking embeddings...")
        # 1. Stack into a single large numpy matrix first (Fast & RAM efficient)
        emb_matrix = np.vstack(all_embeddings)

        print(f"Creating DataFrame from matrix shape {emb_matrix.shape}...")
        # 2. Create DataFrame directly from matrix
        df = pd.DataFrame(emb_matrix)

        # 3. Attach MultiIndex
        df.index = pd.MultiIndex.from_tuples(all_indices, names=['eid', 'visit_index'])

        # 4. Rename Columns
        df.columns = [f"dino_{i}" for i in range(df.shape[1])]

        print(f"Saving to {OUTPUT_FILE}...")
        df.to_pickle(OUTPUT_FILE)
        print("Done!")
    else:
        print("No embeddings extracted. Check dataset path or errors.")

if __name__ == "__main__":
    extract_features()