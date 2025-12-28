import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.001):
    def lr_lambda(current_step):
        # Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine Decay Phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        # Standard cosine term: starts at 1.0, ends at 0.0
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        # Scale to range [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def insure_indices_format(df):
    df['RegistrationCode'] = df['RegistrationCode'].astype(str).apply(lambda x: f"10K_{x}" if not x.startswith("10K_") else x)
    df['research_stage']  = df['research_stage'].replace('00_00_visit', 'baseline')
    return df


def prepare_combined_manifest(manifest, crops_manifest):
    """
    Merges the Full Body manifest with the Crops manifest.
    """
    print("Loading Manifests...")
    fb_df = pd.read_pickle(manifest)

    # Ensure ID formatting matches
    fb_df = insure_indices_format(fb_df)

    # Load Crops
    crops_df = pd.read_pickle(crops_manifest)
    crops_df = insure_indices_format(crops_df)

    # Group Crops by Visit
    crops_grouped = crops_df.groupby(['RegistrationCode', 'research_stage'])['FilePath'].apply(list).to_dict()

    # Add crops column to full body df
    def get_crops(row):
        key = (row['RegistrationCode'], row['research_stage'])
        return crops_grouped.get(key, [])

    fb_df['Crop_Paths'] = fb_df.apply(get_crops, axis=1)
    return fb_df