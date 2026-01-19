import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# Import your modules
from Dataset import DEXADataset
from dinov3_model import DINOv3
from torchvision.transforms import v2

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
FULL_BODY_SCANS_CACHE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl"
OUTPUT_FILE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dino_zeroshot_small_fixed_normalization.pkl"

# Set to None if you want to use the vanilla pre-trained DINO weights.
CHECKPOINT_PATH = None

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def get_dataset():
    #  Load the sample list
    if not os.path.exists(FULL_BODY_SCANS_CACHE):
        print("Please run dexa_dataset.py first to generate the sample cache.")
        return None

    print(f"Loading samples from {FULL_BODY_SCANS_CACHE}...")
    df = pd.read_pickle(FULL_BODY_SCANS_CACHE)

    transform = make_transform(resize_size=256)
    dataset = DEXADataset(
        df,
        transform=transform,
    )
    return dataset

def extract_features():
    dataset = get_dataset()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)


    # Initialize Model
    print("Initializing DINO backbone...")
    # We use the same wrapper, but we'll access 'model.backbone' directly
    model = DINOv3(model_name='vit_small_patch16_dinov3.lvd1689m')

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading fine-tuned weights from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Using vanilla pre-trained DINO weights.")

    model = model.to(DEVICE)
    model.eval()

    all_embeddings = []
    all_ids = []
    all_visits = []

    print("Starting extraction...")
    with torch.no_grad():
        for images, sample_ids in tqdm(dataloader):
            images = images.to(DEVICE)
            # Extract features from the backbone
            embeddings = model.backbone(images)

            # Move to CPU and numpy
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(sample_ids[0])
            all_visits.extend(sample_ids[1])

    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"Extraction complete. Shape: {all_embeddings.shape}")

    # Save to DataFrame - wide format
    output_df = pd.DataFrame(all_embeddings)
    output_df.insert(0, 'RegistrationCode', all_ids)
    output_df.insert(1, 'research_stage', all_visits)



    print(f"Saving to {OUTPUT_FILE}...")
    output_df.to_pickle(OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    extract_features()