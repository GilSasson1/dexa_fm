import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

# --- IMPORTS FROM YOUR CODE ---
from model import LeJEPA_Encoder, SIGReg # Assuming these are in model.py
from augmentations import DEXA_MEAN, DEXA_STD, PadToRatio

# --- CONFIGURATION ---
MODEL_NAME = 'vit_large_patch16_224'
MODEL_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/checkpoints/lejepa_dexa/best_model_no_batchnorm_test.pth"
OUTPUT_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/lejepa/residuals_analysis.pkl"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANIFEST_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/dxa_total_body_manifest.pkl"

# MODIFIED SIGREG FOR EVALUATION
class SigRegEvaluator(SIGReg):
    """
    Inherits from your SIGReg but computes loss PER SAMPLE (no batch averaging).
    Also fixes the projection matrix A to ensure scores are consistent across batches.
    """
    def __init__(self, num_slices=2048, embed_dim=768, seed=42):
        super().__init__(num_slices=num_slices)

        # FIX THE RANDOM PROJECTION MATRIX 'A'
        # In training, A is random every step. In eval, it must be constant
        # so sample A and sample B are judged by the same ruler.
        torch.manual_seed(seed)
        A = torch.randn(embed_dim, self.num_slices)
        A = A.div_(A.norm(dim=0, keepdim=True) + 1e-6)
        self.register_buffer('fixed_A', A) # Saved as part of module

    def forward_per_sample(self, z):
        """
        z: (Batch, Dim)
        Returns: (Batch,) vector of losses
        """
        B, D = z.shape

        # Use the FIXED projection matrix
        # (Batch, Dim) @ (Dim, Slices) -> (Batch, Slices)
        z_proj = z @ self.fixed_A

        # Compute Empirical Characteristic Function
        # val: (Batch, Slices, Knots)
        # self.t is (1, 1, Knots) -> Broadcaster works automatically
        val = z_proj.unsqueeze(-1) * self.t

        # --- KEY CHANGE: DO NOT MEAN OVER BATCH (dim 0) ---
        # Calculate deviation from target for EACH sample
        # target_cf is (1, Knots)

        # Real part error: (cos(val) - target)^2
        diff_real = (val.cos() - self.target_cf).square()
        # Imag part error: sin(val)^2
        diff_imag = val.sin().square()

        # Total diff: (Batch, Slices, Knots)
        diff = diff_real + diff_imag

        # Integrate over Knots (last dim)
        # self.weights: (Knots,)
        # (Batch, Slices, Knots) @ (Knots,) -> (Batch, Slices)
        loss_integrated = diff @ self.weights

        # Mean over Slices (dim 1) -> (Batch,)
        loss_per_sample = loss_integrated.mean(dim=1)

        return loss_per_sample

# DATASET & TRANSFORMS
class TwoViewTransform:
    """Returns two views of the same image to calculate invariance error."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        v1 = self.transform(x)
        v2 = self.transform(x) # Different random augmentations applied
        return v1, v2

class DEXAExtractionDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.manifest = manifest_df
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def load_image(self, path):
        # ... (Same as your provided code) ...
        try:
            if path.endswith('.npy'):
                arr = np.load(path).astype(np.float32)
                # Quick Norm to uint8 for PIL-like handling if needed,
                # or keep float if your pipeline expects it.
                # Keeping your logic:
                if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
                mn, mx = arr.min(), arr.max()
                if mx - mn > 0: arr = (arr - mn) / (mx - mn)
                return (arr * 255).astype(np.uint8)
            else:
                return np.array(Image.open(path).convert('RGB'))
        except Exception:
            return np.zeros((100,100,3), dtype=np.uint8) # Fallback

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        reg_code = row['RegistrationCode']
        visit_id = row['research_stage'] if 'research_stage' in row else 'baseline'

        # Load Composite (simplifying for brevity, use your full loading logic)
        comp_img = self.load_image(row['Path_Composite'])
        bone_img = self.load_image(row['Path_Bone'])
        tissue_img = self.load_image(row['Path_Tissue'])

        # Resize to match
        h, w = comp_img.shape[:2]
        bone_img = np.array(Image.fromarray(bone_img).resize((w, h)))
        tissue_img = np.array(Image.fromarray(tissue_img).resize((w, h)))

        # Stack
        img_array = np.stack([bone_img[..., 0], tissue_img[..., 0], comp_img[..., 0]], axis=0)
        img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0

        # Apply Transform (Returns (v1, v2))
        if self.transform:
            v1, v2 = self.transform(img_tensor)
        else:
            v1, v2 = img_tensor, img_tensor

        return v1, v2, reg_code, visit_id

#  MAIN EXTRACTION LOOP
def main():
    print(f"--- Calculating Residuals on {DEVICE} ---")

    model = LeJEPA_Encoder(MODEL_NAME, img_size=(224, 224), proj_out_dim=128).to(DEVICE)
    model.eval()

    if os.path.exists(MODEL_PATH):
        print(f"Loading weights...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # Clean weights if needed (remove 'module.' prefix)
        state_dict = checkpoint['encoder'] if 'encoder' in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found")

    # SigReg Setup
    # We need to know the embedding dimension coming out of the projector.
    sigreg_eval = SigRegEvaluator(num_slices=2048, embed_dim=128).to(DEVICE)

    # Data Setup
    print("Loading Manifest...")
    manifest = pd.read_pickle(MANIFEST_PATH) # or read_csv

    # We need slight augmentation to measure invariance error
    # If we use 100% identical crops, PredLoss will be 0.
    eval_transform = TwoViewTransform(transforms.Compose([
        PadToRatio(target_ratio=2.0),
        # Add a very slight random crop or flip to test robustness
        transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0), ratio=(0.5, 0.5)),
        transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD),
    ]))

    dataset = DEXAExtractionDataset(manifest, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Loop
    results = []

    # Lambda for total loss (from your training config)
    LAMBDA_SIGREG = 0.3

    print("Running Inference...")
    with torch.no_grad():
        for v1, v2, reg_codes, visit_ids in tqdm(dataloader):
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)

            # Forward Pass
            # We need the PROJECTIONS, not just features
            _, proj1 = model(v1)
            _, proj2 = model(v2)

            # 1. Prediction Loss (Invariance)
            # MSE between projections of view 1 and view 2
            # Shape: (Batch,)
            pred_loss = F.mse_loss(proj1, proj2, reduction='none').mean(dim=1)

            # 2. SigReg Loss (Regularization)
            # How "weird" is the projection distribution?
            # We average the cost for v1 and v2 to get a stable estimate
            sig_loss1 = sigreg_eval.forward_per_sample(proj1)
            sig_loss2 = sigreg_eval.forward_per_sample(proj2)
            sig_loss = (sig_loss1 + sig_loss2) / 2

            # 3. Total Loss
            total_loss = pred_loss + (LAMBDA_SIGREG * sig_loss)

            # Store
            p_np = pred_loss.cpu().numpy()
            s_np = sig_loss.cpu().numpy()
            t_np = total_loss.cpu().numpy()

            for i in range(len(reg_codes)):
                results.append({
                    'RegistrationCode': reg_codes[i],
                    'research_stage': visit_ids[i],
                    'pred_loss': p_np[i],
                    'sigreg_loss': s_np[i],
                    'total_loss': t_np[i]
                })

    # E. Save
    df_res = pd.DataFrame(results)
    df_res.set_index(['RegistrationCode', 'research_stage'], inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_res.to_pickle(OUTPUT_PATH)
    print(f"Saved residuals to {OUTPUT_PATH}")
    print(df_res.describe())

if __name__ == "__main__":
    main()