import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
# Import your modules
from DEXA.dexa_fm.Dino_Dataset import DEXADataset
from dinov3_model import DINOv3
from torchvision.transforms import v2
import timm
from torchvision import transforms

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
FULL_BODY_SCANS_CACHE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl"
OUTPUT_FILE = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dino_zeroshot_small_fixed_normalization.pkl"

# Set to None if you want to use the vanilla pre-trained DINO weights.
CHECKPOINT_PATH = None

def get_dinov3_transform(model_name='vit_small_patch16_dinov3.lvd1689m'):
    """
    Get the normalization transform for DINOv3.
    
    Since DEXADataset already:
    - Loads images as float32 tensors scaled to [0,1]
    - Resizes to 256x256 with BICUBIC interpolation
    
    We only need to apply the ImageNet normalization here.
    
    DINOv3 uses standard ImageNet normalization:
    - mean = (0.485, 0.456, 0.406)
    - std = (0.229, 0.224, 0.225)
    """
    
    # Only normalization - Dataset handles loading, resizing
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    
    print(f"DINOv3 Transform: Normalize with ImageNet mean/std (Dataset handles resize to 256x256 BICUBIC)")
    return normalize

# Model name used for both transform and model loading
MODEL_NAME = 'vit_small_patch16_dinov3.lvd1689m'

def get_dataset():
    #  Load the sample list
    if not os.path.exists(FULL_BODY_SCANS_CACHE):
        print("Please run dexa_dataset.py first to generate the sample cache.")
        return None

    print(f"Loading samples from {FULL_BODY_SCANS_CACHE}...")
    df = pd.read_pickle(FULL_BODY_SCANS_CACHE)

    # Use official DINOv3 preprocessing from timm
    transform = get_dinov3_transform(model_name=MODEL_NAME)
    dataset = DEXADataset(
        df,
        transform=transform,
    )
    return dataset

def extract_features():
    dataset = get_dataset()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, prefetch_factor=2)


    # Initialize Model
    print("Initializing DINO backbone...")
    # We use the same wrapper, but we'll access 'model.backbone' directly
    model = DINOv3(model_name=MODEL_NAME)

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading fine-tuned weights from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Using vanilla pre-trained DINO weights.")

    model = model.to(DEVICE)
    model.eval()

    all_embeddings_bone = []
    all_embeddings_tissue = []
    all_ids = []
    all_visits = []

    print("Starting extraction...")
    # Use inference_mode + autocast to match official DINOv3 inference settings
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for (bone_images, tissue_images), sample_ids in tqdm(dataloader):
                bone_images = bone_images.to(DEVICE)
                tissue_images = tissue_images.to(DEVICE)
                # Extract features from the backbone
                embeddings_tissue = model.backbone(tissue_images)
                embeddings_bone = model.backbone(bone_images)

                # Move to CPU and numpy
                all_embeddings_bone.append(embeddings_bone.cpu().numpy())
                all_embeddings_tissue.append(embeddings_tissue.cpu().numpy())
                all_ids.extend(sample_ids[0])
                all_visits.extend(sample_ids[1])

    # Concatenate all batches
    all_embeddings_bone = np.concatenate(all_embeddings_bone, axis=0)
    all_embeddings_tissue = np.concatenate(all_embeddings_tissue, axis=0)

    print(f"Extraction complete. Shape: {all_embeddings_bone.shape}")

    # Save to DataFrame - wide format
    output_df_bone  = pd.DataFrame(all_embeddings_bone)
    output_df_tissue  = pd.DataFrame(all_embeddings_tissue)
    output_df_bone.insert(0, 'RegistrationCode', all_ids)
    output_df_bone.insert(1, 'research_stage', all_visits)

    output_df_tissue.insert(0, 'RegistrationCode', all_ids)
    output_df_tissue.insert(1, 'research_stage', all_visits)

    print(f"Saving to {OUTPUT_FILE}...")
    output_df_bone.to_pickle(f"{OUTPUT_FILE}_bone.pkl")
    output_df_tissue.to_pickle(f"{OUTPUT_FILE}_tissue.pkl")
    print("Done!")

if __name__ == "__main__":
    extract_features()