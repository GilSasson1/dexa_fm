import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import random
from lejepa_dataset import LeJEPAHDF5Dataset

# --- CONFIGURATION ---
HDF5_PATH = '/net/mraid20/export/jasmine/UKBB_DATA/ukbb_dexa_dataset_v3.h5'

class StatsTransform:
    def __init__(self):
        self.global_trans = transforms.Compose([
            transforms.Resize((384, 128), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ])
    def local_trans(self, x): return self.global_trans(x)
    def synthetic_local_trans(self, x): return self.global_trans(x)

def compute_mean_std(batch_size=128, num_workers=8, subset_fraction=1.0):
    
    # --- NEW: Extract and Subset Keys ---
    print(f"Scanning HDF5 keys from {HDF5_PATH}...")
    with h5py.File(HDF5_PATH, 'r') as f:
        all_keys = list(f.keys())
        
    if subset_fraction < 1.0:
        random.seed(42) # Ensure we get the same random subset if we run it twice
        num_to_keep = int(len(all_keys) * subset_fraction)
        selected_keys = random.sample(all_keys, num_to_keep)
        print(f"Subset Active: Using {subset_fraction*100}% of data ({len(selected_keys)}/{len(all_keys)} scans)...")
    else:
        selected_keys = all_keys
        print(f"Using all {len(selected_keys)} scans...")

    # Initialize Dataset with our subset
    dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=selected_keys, 
        targets_df=None,
        transform=StatsTransform(),
        n_global=2, 
        n_local=0
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    # FIRST PASS (MEAN)
    for views, _ in tqdm(loader, desc="Pass 1/2: Mean"):
        images = torch.cat([views[0], views[1]], dim=0) 
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    
    # SECOND PASS (STD)
    for views, _ in tqdm(loader, desc="Pass 2/2: Std"):
        images = torch.cat([views[0], views[1]], dim=0) 
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        mean_reshaped = mean.view(1, 3, 1)
        std += ((images - mean_reshaped) ** 2).mean(2).sum(0)

    std = torch.sqrt(std / total_samples)

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    # For UKBB, 0.1 (10%) or 0.2 (20%) is more than enough to get highly accurate stats!
    calculated_mean, calculated_std = compute_mean_std(batch_size=128, num_workers=8, subset_fraction=0.2)
    
    print("\n--- RESULTS ---")
    print(f"Dataset: {HDF5_PATH}")
    print(f"MEAN = {calculated_mean}")
    print(f"STD  = {calculated_std}")