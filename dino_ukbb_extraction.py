import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dinov3_model import DINOv3

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
NPY_DIR = '/net/mraid20/export/jasmine/UKBB_DATA/preprocessed_dexa_images/fullbody'
OUTPUT_FILE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/ukbb_dexa_embeddings_dino.pkl"

# DINO SPECS (Matching your HPP Script)
HEIGHT = 224
WIDTH = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- DATASET CLASS ---
class UKBB_Npy_Dataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.files = glob(os.path.join(npy_dir, '*.npy'))
        self.transform = transform
        print(f"Found {len(self.files)} processed .npy files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)

        # LOGIC: Extract ID and Visit
        # Expects: "1234567_2_0_fullbody.npy" -> eid=1234567, visit=2_0
        parts = filename.split('_')
        patient_id = parts[0]

        # Safe visit ID extraction
        if len(parts) >= 3 and parts[1].isdigit():
            visit_id = f"{parts[1]}_{parts[2]}"
        else:
            visit_id = "2_0" # Default

        try:
            # 1. Load Data (uint8, Shape: 3, H, W)
            img_arr = np.load(path)

            # 2. Convert to Tensor (Float 0.0 - 1.0)
            img_tensor = torch.from_numpy(img_arr).float() / 255.0

            # 3. Apply Transforms
            if self.transform:
                img_tensor = self.transform(img_tensor)

            return img_tensor, patient_id, visit_id

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return torch.zeros((3, HEIGHT, WIDTH)), "error", "error"

def extract_features():
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    dataset = UKBB_Npy_Dataset(NPY_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("Initializing DINO backbone...")
    model = DINOv3(model_name='vit_small_patch16_dinov3.lvd1689m')
    model = model.to(DEVICE)
    model.eval()

    # --- MEMORY OPTIMIZATION START ---
    # Instead of a dict, we use lists for metadata and a pre-allocated list for arrays
    # (or you can use h5py if RAM is extremely tight, but this should work for 70k)

    all_embeddings = []
    all_indices = [] # Will hold tuples of (eid, visit_index)

    print("Starting extraction...")
    with torch.no_grad():
        for images, pids, vids in tqdm(dataloader):
            images = images.to(DEVICE)

            try:
                embeddings = model.backbone(images)
            except AttributeError:
                embeddings = model(images)

            # Move to CPU numpy and Append to list
            # Note: Appending numpy arrays to a list is cheap.
            # Converting that list to a DataFrame is the expensive part if done wrong.
            batch_emb = embeddings.cpu().numpy()

            for i, (pid, vid) in enumerate(zip(pids, vids)):
                if pid != "error":
                    all_embeddings.append(batch_emb[i])
                    all_indices.append((pid, vid))

    print(f"Extraction complete. Found {len(all_embeddings)} embeddings.")

    # --- EFFICIENT DATAFRAME CREATION ---
    print("Stacking embeddings...")
    # 1. Stack into a single large numpy matrix first (Fast & RAM efficient)
    emb_matrix = np.vstack(all_embeddings)

    print(f"Creating DataFrame from matrix shape {emb_matrix.shape}...")
    # 2. Create DataFrame directly from matrix (Avoids intermediate list-of-tuples)
    df = pd.DataFrame(emb_matrix)

    # 3. Attach Index
    df.index = pd.MultiIndex.from_tuples(all_indices, names=['eid', 'visit_index'])

    # 4. Rename Columns
    df.columns = [f"dino_{i}" for i in range(df.shape[1])]

    print(f"Saving to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    extract_features()