import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
from LeJEPA_dataset import LeJEPAHDF5Dataset

HDF5_PATH = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'



stats_transform = transforms.Compose([
    transforms.Resize((384, 128), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),

])


def compute_mean_std(dataset_class, batch_size=32, num_workers=4):

    dataset = dataset_class(
        hdf5_path=HDF5_PATH,
        transform=stats_transform,
        n_global=1,
        n_local=0
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = 0.0
    std = 0.0
    total_samples = 0

    print("Computing Mean and Std...")

    # FIRST PASS (MEAN)
    for batch in tqdm(loader):
        # The dataset now returns JUST the image tensor, no tuple/list unpacking needed
        # But just in case:
        if isinstance(batch, list) or isinstance(batch, tuple):
            images = batch[0] # Handle case if you return (image, target)
        else:
            images = batch

        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples

    # SECOND PASS (STD)
    for batch in tqdm(loader):
        if isinstance(batch, list) or isinstance(batch, tuple):
            images = batch[0]
        else:
            images = batch

        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean_reshaped = mean.view(1, 3, 1)
        std += ((images - mean_reshaped) ** 2).mean(2).sum(0)

    std = torch.sqrt(std / total_samples)

    return mean.tolist(), std.tolist()

# --- 3. Usage ---
if __name__ == "__main__":

    calculated_mean, calculated_std = compute_mean_std(LeJEPAHDF5Dataset, batch_size=64, num_workers=4)
    print(f"DEXA_MEAN = {calculated_mean}")
    print(f"DEXA_STD  = {calculated_std}")
    pass