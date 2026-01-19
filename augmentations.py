import torch
from torchvision import transforms

DEXA_MEAN = [0.12973763048648834, 0.2611643970012665, 0.19504129886627197]
DEXA_STD  = [0.22688911855220795, 0.3180334270000458, 0.24720656871795654]

class train_transforms:
    def __init__(self, global_size=(384, 128), local_size=(192, 64)):

        self.normalize = transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD)
        self.intensity_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4)

        # A. Global Transform
        self.global_trans = transforms.Compose([
            transforms.RandomResizedCrop(
                global_size,
                scale=(0.5, 1.0),
                ratio=(0.3, 0.6),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, .1)], p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            self.normalize
        ])

        # B. Local Transform (Real Crops)
        self.local_trans = transforms.Compose([
            transforms.RandomResizedCrop(
                (96, 96),
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),  # Near Square for local scans
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            # Note: No normalize here (handled by Dataset after stacking)
        ])

        # C. Synthetic Local Source (Crops from Full Body)
        self.synthetic_local_trans = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=(0.05, 0.3),
                ratio=(0.3, 0.6),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            self.normalize
        ])

class val_transforms:
    def __init__(self, global_size=(384, 128)):
        self.normalize = transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD)

        self.global_trans = transforms.Compose([
            transforms.Resize(global_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            self.normalize
        ])

        self.local_trans = self.global_trans
        self.synthetic_local_trans = self.global_trans