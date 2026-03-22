import torch
from torchvision import transforms

MEAN = [0.1960878074169159, 0.1960878074169159, 0.1960878074169159]
STD  = [0.2843901515007019, 0.2843901515007019, 0.2843901515007019]


class train_transforms:
    def __init__(self, global_size=(384, 128), local_size=(96, 96), mean=MEAN, std=STD):
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.intensity_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4)

        # --- A. Global Transform (Tall & Thin) ---
        # Input: ~750x272 (AR ~0.36) -> Target: 384x128 (AR 0.33)
        self.global_trans = transforms.Compose([
            transforms.RandomResizedCrop(
                global_size,
                scale=(0.5, 1.0),
                # Aspect Ratio (W/H). Target is 0.33. We allow slight variation around that.
                ratio=(0.25, 0.45),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomRotation(degrees=10), # Reduced rotation for tall images
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, .1)], p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            transforms.ToTensor(),
            self.normalize
        ])

        # --- B. Local Transform (Real High-Res Crops) ---
        # Input: ~850x640 (AR 0.75) or ~600x680 (AR 1.1) -> Target: Square 96x96
        self.local_trans = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=(0.6, 1.0), # Zoom in on the high-res detail
                # Handles both the tall spine crops and wide hip crops
                ratio=(0.7, 1.3),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            transforms.ToTensor(),
            self.normalize
        ])

        # --- C. Synthetic Local (Fallback) ---
        # Input: Full Body (Tall) -> Target: Local Patch
        # Since Full Body is low res, we must crop small areas to simulate local views
        self.synthetic_local_trans = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=(0.1, 0.3), # Grab small patches from full body
                ratio=(0.7, 1.3), # Force square-ish output
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            transforms.ToTensor(),
            self.normalize
        ])

class val_transforms:
    def __init__(self, global_size=(384, 128), mean=MEAN, std=STD):
        # Same Mean/Std as training
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.global_trans = transforms.Compose([
            # Deterministic Resize (No RandomResizedCrop)
            transforms.Resize(
                global_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.ToTensor(),
            self.normalize
        ])

    # Validation usually doesn't need local views, but if called:
    def local_trans(self, x):
        return self.global_trans(x)

    def synthetic_local_trans(self, x):
        return self.global_trans(x)