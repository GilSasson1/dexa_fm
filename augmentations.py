import torch
from torchvision import transforms


DEXA_MEAN = [0.12394659966230392, 0.2626885771751404, 0.3075794577598572]
DEXA_STD  = [0.21822425723075867, 0.31778785586357117, 0.3350508213043213]

class LeJEPATransformConfig:
    """Holds the transformation pipelines"""
    def __init__(self, global_size=224, local_size=96):
        self.normalize = transforms.Normalize(mean=DEXA_MEAN, std=DEXA_STD)
        self.to_float = transforms.ConvertImageDtype(torch.float)

        self.fb_pre = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.intensity_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2)

        # B. Global Transform
        self.global_trans = transforms.Compose([
            self.to_float,
            transforms.RandomResizedCrop(global_size, scale=(0.3, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, .1)], p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])

        # C. Local Transform
        self.local_trans = transforms.Compose([
            self.to_float,
            transforms.RandomResizedCrop(local_size, scale=(0.25, 1.0), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])

        # D. Synthetic Local Source Transform
        self.synthetic_local_trans = transforms.Compose([
            self.to_float,
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.3), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.intensity_aug], p=0.8),
            self.normalize
        ])
