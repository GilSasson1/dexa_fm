import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CUSTOM MODULES ---
from model import LeJEPA_Encoder
import sys
# 1. Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory
parent_dir = os.path.dirname(current_dir)
# 3. Add parent directory to sys.path
sys.path.append(parent_dir)
# 4. Now you can import
from Dataset import DEXADataset, FULL_BODY_SCANS_CACHE

# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# STATS
DEXA_MEAN = [0.12394659966230392, 0.2626885771751404, 0.3075794577598572]
DEXA_STD  = [0.21822425723075867, 0.31778785586357117, 0.3350508213043213]

# --- CONFIG ---
# Path to the specific checkpoint you want to load
CHECKPOINT_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/best_model_vit_large_patch16_224_0.05_250_0.0001_moreviews.pth"
OUTPUT_DIR = "/net/mraid20/export/genie/LabData/Analyses/gilsa/training_evolution/lejepa_att_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image Config
IMG_HEIGHT = 416
IMG_WIDTH = 128   # Must be divisible by PATCH_SIZE
PATCH_SIZE = 16 # ViT Patch Size
BATCH_SIZE = 16
MAX_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# 1. Visualization Helper
# ---------------------------------------------------------
def force_visible_range(img_data):
    img_data = img_data.astype(np.float32)
    if img_data.max() == img_data.min(): return np.zeros_like(img_data)
    return (img_data - img_data.min()) / (img_data.max() - img_data.min())

def generate_and_save_maps(model, loader, output_dir, max_samples=10):
    model.eval()

    # 1. Disable Fused Attention (CRITICAL for hooks to work)
    # If fused_attn is True, PyTorch uses a C++ kernel that skips python hooks.
    print("Disabling Fused Attention for visualization...")
    for block in model.backbone.blocks:
        if hasattr(block.attn, 'fused_attn'):
            block.attn.fused_attn = False

    # 2. Register Hook
    attentions = []
    def get_attention(module, input, output):
        attentions.append(input[0])

    # Hook onto the dropout layer of the last block's attention
    target_layer = model.backbone.blocks[23].attn.attn_drop
    hook = target_layer.register_forward_hook(get_attention)

    saved_count = 0
    print(f"Generating maps for first {max_samples} samples...")

    with torch.no_grad():
        for batch_idx, (imgs, subject_ids) in enumerate(loader):
            attentions = [] # Clear buffer
            imgs = imgs.to(DEVICE)

            _ = model(imgs)

            if not attentions:
                print("Error: No attention captured. The model might be using functional SDPA (Flash Attention).")
                print("Try setting 'use_fused=False' in your model config or check if 'attn_drop' is actually called.")
                break

            attn_batch = attentions[0] # [Batch, Heads, Tokens, Tokens]

            for i in range(imgs.shape[0]):
                if saved_count >= max_samples:
                    hook.remove()
                    return

                # A. Prepare Image
                raw_img = imgs[i].cpu().numpy().transpose(1, 2, 0)
                bg_image = force_visible_range(raw_img[:, :, 2]) # Grayscale

                # B. Process Attention
                avg_attn = attn_batch[i].mean(dim=0) # Average over heads

                # C. Calculate Rectangular Grid
                w_featmap = imgs.shape[3] // PATCH_SIZE
                h_featmap = imgs.shape[2] // PATCH_SIZE
                num_patches = w_featmap * h_featmap

                # D. Extract Patches (Skip CLS/Registers)
                # We take the last N tokens, which corresponds to the spatial grid
                cls_attn = avg_attn[0, -num_patches:]

                try:
                    heatmap = cls_attn.reshape(h_featmap, w_featmap).cpu().numpy()
                except RuntimeError:
                    print(f"Reshape Error! Got {len(cls_attn)} tokens, needed {num_patches}.")
                    continue

                heatmap = force_visible_range(heatmap)

                # E. Overlay & Save
                pil_map = Image.fromarray((heatmap * 255).astype(np.uint8))
                pil_map = pil_map.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.BILINEAR)
                overlay = np.array(pil_map).astype(np.float32) / 255.0

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(bg_image, cmap='gray', vmin=0, vmax=1)
                ax.imshow(overlay, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f"Subject ID: {subject_ids[0][i]}", fontsize=12)
                ax.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"lejepa_sample_{saved_count}.png"))
                plt.close(fig)

                saved_count += 1

    hook.remove()
    print("Done.")

# ---------------------------------------------------------
# 2. Main Execution
# ---------------------------------------------------------
print("Loading Data...")
manifest_df = pd.read_pickle(FULL_BODY_SCANS_CACHE)
manifest_df['RegistrationCode'] = manifest_df['RegistrationCode'].apply(
    lambda x: f"10K_{x}" if not x.startswith("10K_") else x
)

subset_df = manifest_df.iloc[:20]

val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD)
])

dataset = DEXADataset(subset_df, transform=val_transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL LOADING (UPDATED) ---
print(f"Loading Lejepa from {CHECKPOINT_PATH}...")
model = LeJEPA_Encoder('vit_large_patch16_224', pretrained=False, proj_out_dim=128)
model.to(DEVICE)

# 2. Load Checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)


if 'encoder' in checkpoint:
    state_dict = checkpoint['encoder']
else:
    # Fallback if the user changes checkpoints later
    state_dict = checkpoint

# 4. Clean 'module.' (Just in case)
# Even though your print didn't show it, this is safe to keep.
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:]
    new_state_dict[k] = v

# 5. Load
msg = model.load_state_dict(new_state_dict, strict=False)

print("Load Results:")
print(f"  Missing Keys: {len(msg.missing_keys)}")
# We expect missing keys for the 'head' (fc layers), but NOT for the backbone.
if len(msg.missing_keys) > 0:
    print(f"  Example Missing: {msg.missing_keys[:3]}")

# 6. Run Visualization
generate_and_save_maps(model, loader, OUTPUT_DIR, max_samples=MAX_SAMPLES)