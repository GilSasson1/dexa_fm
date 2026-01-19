import os
import h5py
import pydicom
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import hashlib

# --- CONFIGURATION ---
DICOM_ROOT = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/dicom/"
OUTPUT_DIR = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/"
HDF5_NAME = "dxa_dataset.h5"

# --- GEOMETRIC CONSTANTS
AR_MIN = 2.2
AR_MAX = 4.0

# --- HELPER FUNCTIONS ---

def _8bit_cast(arr):
    """
    Fastest possible conversion.
    Assumes input is already roughly 8-bit range.
    """
    if arr is None: return None

    # If it's already uint8, do nothing (Speed up!)
    if arr.dtype == np.uint8:
        return arr

    # If it's int8 (-128 to 127), shift it to 0-255
    if arr.dtype == np.int8:
        return (arr.astype(np.int16) + 128).astype(np.uint8)

    # Fallback for unexpected types: Simple Clip & Cast
    # We assume values are already in 0-255 range based on your input.
    return np.clip(arr, 0, 255).astype(np.uint8)

def classify_dicom(ds, filename):
    """classify scan types by Geometry"""
    if 'file4' in filename: return None

    rows = float(ds.get('Rows', 0))
    cols = float(ds.get('Columns', 0))

    if rows < 50 or cols < 50: return None

    aspect_ratio = rows / cols

    # Logic: Is it a tall strip?
    is_full_body = (AR_MIN < aspect_ratio < AR_MAX)

    if is_full_body:
        return 'fullbody'
    return 'crop'

def get_dicom_meta(path):
    try:
        # stop_before_pixels is CRITICAL for speed
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        img_type = classify_dicom(ds, os.path.basename(path))
        if img_type is None: return None

        return {
            'path': path,
            'type': img_type,
            'InstanceNumber': int(ds.get('InstanceNumber', 0)),
        }
    except:
        return None

def format_ids(subj, visit):
    """Bakes the 10K logic and baseline logic into the strings"""
    # 1. Subject ID
    s = str(subj)
    if not s.startswith("10K_"):
        s = f"10K_{s}"

    # 2. Visit ID
    v = str(visit)
    if v == '00_00_visit':
        v = 'baseline'

    return s, v

# --- MAIN PIPELINE ---

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hdf5_path = os.path.join(OUTPUT_DIR, HDF5_NAME)

    # 1. SCAN
    print("STEP 1: Scanning headers...")
    groups = defaultdict(list)
    files = [os.path.join(r, f) for r, d, fl in os.walk(DICOM_ROOT) for f in fl if f.endswith('.dcm') and 'file4' not in f]

    for f in tqdm(files, desc="Classifying"):
        meta = get_dicom_meta(f)
        if meta is None: continue
        parts = f.split(os.sep)
        try:
            visit_id = parts[-3]
            subj_id = parts[-4]
            groups[(subj_id, visit_id)].append(meta)
        except: continue

    # 2. WRITE WITH FORMATTING
    print(f"STEP 2: Packing {len(groups)} visits...")

    with h5py.File(hdf5_path, 'w', libver='latest') as hf:

        for (raw_subj, raw_visit), items in tqdm(groups.items()):

            # --- APPLY FORMATTING HERE ---
            fmt_subj, fmt_visit = format_ids(raw_subj, raw_visit)

            # Use the clean IDs for the Group Key too!
            # This makes splitting in main.py much easier.
            group_key = f"{fmt_subj}_{fmt_visit}"

            # Check duplicates (if multiple raw folders map to same clean ID)
            if group_key in hf:
                continue

            full_bodies = [x for x in items if x['type'] == 'fullbody']
            crops = [x for x in items if x['type'] == 'crop']
            if len(full_bodies) < 2: continue

            full_bodies.sort(key=lambda x: x['InstanceNumber'])
            bone_meta = full_bodies[0]
            tissue_meta = full_bodies[-1]

            try:
                # Load & Process
                bone_arr = _8bit_cast(pydicom.dcmread(bone_meta['path']).pixel_array)
                tissue_arr = _8bit_cast(pydicom.dcmread(tissue_meta['path']).pixel_array)
                comp_arr = ((bone_arr.astype(np.float32) + tissue_arr.astype(np.float32)) / 2.0).astype(np.uint8)
                fb_stack = np.stack([bone_arr, tissue_arr, comp_arr], axis=0)

                # Deduplicate Crops
                unique_crops = []
                seen_hashes = set()
                seen_hashes.add(hashlib.md5(bone_arr.tobytes()).hexdigest())
                seen_hashes.add(hashlib.md5(tissue_arr.tobytes()).hexdigest())

                for c in crops:
                    try:
                        c_arr = _8bit_cast(pydicom.dcmread(c['path']).pixel_array)
                        if c_arr is None: continue
                        h = hashlib.md5(c_arr.tobytes()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_crops.append(c_arr)
                    except: pass

                # SAVE
                g = hf.create_group(group_key)

                # --- SAVE CLEAN ATTRIBUTES ---
                g.attrs['RegistrationCode'] = fmt_subj  # Saved as "10K_..."
                g.attrs['research_stage'] = fmt_visit          # Saved as "baseline"
                g.attrs['NumCrops'] = len(unique_crops)

                g.create_dataset('fullbody', data=fb_stack, compression='gzip')
                if len(unique_crops) > 0:
                    c_grp = g.create_group('crops')
                    for i, c_dat in enumerate(unique_crops):
                        c_grp.create_dataset(str(i), data=c_dat, compression='gzip')

            except Exception: continue

    print(f"Done! Formatted dataset at: {hdf5_path}")

if __name__ == "__main__":
    build_dataset()