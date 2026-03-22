import os
import h5py
import pydicom
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import hashlib

# --- CONFIGURATION ---
DICOM_ROOT = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files_orig/dxa/dicom/"
OUTPUT_DIR = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/dxa/"
HDF5_NAME = "dxa_dataset.h5"

AR_MIN = 1.8
AR_MAX = 4.0

def _8bit_cast(arr):
    if arr is None: return None
    if arr.dtype == np.uint8: return arr
    if arr.dtype == np.int8: return (arr.astype(np.int16) + 128).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)

def classify_dicom(ds, filename):
    if 'file4' in filename: return None
    rows, cols = float(ds.get('Rows', 0)), float(ds.get('Columns', 0))
    if AR_MIN < (rows / cols) < AR_MAX: return 'fullbody'
    return 'crop'

def get_dicom_meta(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        img_type = classify_dicom(ds, os.path.basename(path))
        if img_type is None: return None
        return {'path': path, 'type': img_type, 'InstanceNumber': int(ds.get('InstanceNumber', 0))}
    except: return None

def format_ids(subj, visit):
    s = str(subj)
    if not s.startswith("10K_"): s = f"10K_{s}"
    v = str(visit)
    if v == '00_00_visit': v = 'baseline'
    return s, v

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hdf5_path = os.path.join(OUTPUT_DIR, HDF5_NAME)

    # --- NEW: Get a list of already processed scans ---
    existing_keys = set()
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as hf:
            existing_keys = set(hf.keys())
        print(f"Found {len(existing_keys)} existing scans in HDF5. Will skip these.")

    # SCAN
    print("STEP 1: Scanning headers for NEW files...")
    groups = defaultdict(list)
    files = [os.path.join(r, f) for r, d, fl in os.walk(DICOM_ROOT) for f in fl if f.endswith('.dcm') and 'file4' not in f]

    for f in tqdm(files, desc="Classifying"):
        parts = f.split(os.sep)
        try:
            visit_id = parts[-3]
            subj_id = parts[-4]
            visit_date = parts[-2]
            
            # --- Check if we already have this scan BEFORE doing the slow dicom read ---
            fmt_subj, fmt_visit = format_ids(subj_id, visit_id)
            if f"{fmt_subj}_{fmt_visit}" in existing_keys:
                continue 
                
        except: continue

        # Only read the DICOM header if it's a new patient/visit
        meta = get_dicom_meta(f)
        if meta is None: continue
        
        meta['Date'] = visit_date
        groups[(subj_id, visit_id)].append(meta)

    # WRITE WITH FORMATTING
    print(f"STEP 2: Packing {len(groups)} NEW visits...")

    with h5py.File(hdf5_path, 'a', libver='latest') as hf:
        for (raw_subj, raw_visit), items in tqdm(groups.items()):
            fmt_subj, fmt_visit = format_ids(raw_subj, raw_visit)
            group_key = f"{fmt_subj}_{fmt_visit}"
            if group_key in hf: continue

            full_bodies = [x for x in items if x['type'] == 'fullbody']
            crops = [x for x in items if x['type'] == 'crop']
            if len(full_bodies) < 1: 
                print(f"Warning: Skipping {group_key} due to insufficient full body images.")
                continue

            if len(full_bodies) < 1: 
                continue

            full_bodies.sort(key=lambda x: x['InstanceNumber'])

            try:
                # Handle 1 vs 2 scans
                if len(full_bodies) >= 2:
                    # Standard case: We have both
                    bone_meta = full_bodies[0]
                    tissue_meta = full_bodies[-1]
                    bone_arr = _8bit_cast(pydicom.dcmread(bone_meta['path']).pixel_array)
                    tissue_arr = _8bit_cast(pydicom.dcmread(tissue_meta['path']).pixel_array)
                else:
                    # Rescue case: We only have 1 
                    bone_meta = full_bodies[0]
                    bone_arr = _8bit_cast(pydicom.dcmread(bone_meta['path']).pixel_array)
                    tissue_arr = bone_arr

                unique_crops = []
                seen_hashes = {hashlib.md5(bone_arr.tobytes()).hexdigest(), hashlib.md5(tissue_arr.tobytes()).hexdigest()}

                for c in crops:
                    try:
                        c_arr = _8bit_cast(pydicom.dcmread(c['path']).pixel_array)
                        if c_arr is None: continue
                        h = hashlib.md5(c_arr.tobytes()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_crops.append(c_arr)
                    except: pass

                # --- SAVE NEW STRUCTURE ---
                g = hf.create_group(group_key)
                g.attrs['RegistrationCode'] = fmt_subj
                g.attrs['research_stage'] = fmt_visit
                g.attrs['NumCrops'] = len(unique_crops)
                g.attrs['Date'] = items[0]['Date'] if 'Date' in items[0] else ''

                g.create_dataset('bone', data=bone_arr, compression='gzip')
                g.create_dataset('tissue', data=tissue_arr, compression='gzip')
                
                if len(unique_crops) > 0:
                    c_grp = g.create_group('crops')
                    for i, c_dat in enumerate(unique_crops):
                        c_grp.create_dataset(str(i), data=c_dat, compression='gzip')

            except Exception as e: 
                print(f"Error processing {group_key}: {e}")
                continue

    print(f"Done! Formatted dataset at: {hdf5_path}")

if __name__ == "__main__":
    build_dataset()