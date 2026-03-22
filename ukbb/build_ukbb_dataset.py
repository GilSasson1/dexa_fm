import os
import zipfile
import io
import h5py
import pydicom
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm
import hashlib

# --- CONFIGURATION ---
INPUT_DIR = '/net/mraid20/export/jasmine/UKBB_DATA/dexa_images'
OUTPUT_DIR = '/net/mraid20/export/jasmine/UKBB_DATA'
HDF5_NAME = "ukbb_dexa_dataset_v3.h5"

# --- HELPER FUNCTIONS ---

def get_structural_sharpness(img):
    img_f = img.astype(np.float32)
    blurred = gaussian_filter(img_f, sigma=1.0)
    sx = sobel(blurred, axis=0)
    sy = sobel(blurred, axis=1)
    edge_magnitude = np.hypot(sx, sy)
    return np.sum(edge_magnitude)

def process_pixel_array(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    arr = arr * slope + intercept

    max_val = np.percentile(arr, 99.9)
    min_val = arr.min()
    current_range = max_val - min_val

    arr = np.clip(arr, min_val, max_val)
    if current_range > 1e-5:
        arr = (arr - min_val) / current_range
    else:
        arr = np.zeros_like(arr)

    return (arr * 255).astype(np.uint8)

def process_zip_content(zip_path):
    # --- CHANGED: Use a flat list instead of shape_groups ---
    full_body_candidates = []
    crop_candidates = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for f in z.namelist():
                if not f.endswith('.dcm'): continue

                with z.open(f) as dcm_f:
                    try:
                        ds = pydicom.dcmread(io.BytesIO(dcm_f.read()), force=True)

                        # --- 1. GLOBAL QUALITY CHECKS ---
                        if getattr(ds, 'PhotometricInterpretation', 'UNKNOWN') == 'MONOCHROME1':
                            continue 

                        rows = getattr(ds, 'Rows', 0)
                        cols = getattr(ds, 'Columns', 0)
                        if rows < 50 or cols < 50: continue

                        # --- 2. CLASSIFY ---
                        protocol = str(getattr(ds, 'ProtocolName', '')).upper()
                        body_part = str(getattr(ds, 'BodyPartExamined', '')).upper()
                        
                        valid_keywords = ['TOTAL', 'WHOLE', 'BODY', 'COMPOSITION']
                        is_full_body = any(k in protocol for k in valid_keywords) or \
                                       any(k in body_part for k in valid_keywords)

                        if 'SPINE' in protocol or 'KNEE' in protocol or 'FEMUR' in protocol:
                            is_full_body = False

                        # --- 3. PROCESS IMAGE ---
                        img_arr = process_pixel_array(ds)

                        # --- 4. VISUAL BACKGROUND CHECK ---
                        h, w = img_arr.shape
                        p = 20 if h > 100 else 5
                        corners = [
                            img_arr[0:p, 0:p], img_arr[0:p, -p:], 
                            img_arr[-p:, 0:p], img_arr[-p:, -p:]
                        ]
                        avg_corner = np.mean([c.mean() for c in corners])

                        if avg_corner > 50: continue 

                        # --- 5. ROUTING ---
                        sharpness = get_structural_sharpness(img_arr)
                        fill_percentage = np.mean(img_arr > 10) 

                        if is_full_body:
                            item = {
                                'img': img_arr,
                                'score': sharpness,
                                'fill': fill_percentage,
                                'area': rows * cols # <-- Added to help sort by size later
                            }
                            full_body_candidates.append(item)
                        else:
                            crop_candidates.append(img_arr)

                    except Exception:
                        continue

        return full_body_candidates, crop_candidates

    except Exception:
        return None, None

# --- MAIN PIPELINE ---

def build_dataset():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    hdf5_path = os.path.join(OUTPUT_DIR, HDF5_NAME)

    zips = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.zip')]
    print(f"Found {len(zips)} zip files.")

    # Using 'w' to rebuild the file with the new schema
    with h5py.File(hdf5_path, 'w', libver='latest') as hf:
        count = 0
        for z_path in tqdm(zips):
            filename = os.path.basename(z_path)
            try:
                parts = filename.split('_')
                patient_id = parts[0]
                visit_id = parts[2]
            except:
                continue

            group_key = f"{patient_id}_{visit_id}"
            if group_key in hf: continue

            # --- CHANGED: Unpack list directly ---
            full_bodies, raw_crops = process_zip_content(z_path)
            if not full_bodies or len(full_bodies) < 2: 
                continue

            # Sort by area (largest images first) and grab the top 2
            full_bodies.sort(key=lambda x: x['area'], reverse=True)
            img_a = full_bodies[0]
            img_b = full_bodies[1]

            # Bone vs Tissue Logic (Kept exactly the same)
            if img_a['fill'] < img_b['fill']:
                bone = img_a['img']
                tissue = img_b['img']
            else:
                bone = img_b['img']
                tissue = img_a['img']
            
            if abs(img_a['fill'] - img_b['fill']) < 0.05:
                if img_a['score'] > img_b['score']:
                    bone, tissue = img_a['img'], img_b['img']
                else:
                    bone, tissue = img_b['img'], img_a['img']

            # Deduplicate Crops
            unique_crops = []
            seen_hashes = {hashlib.md5(bone.tobytes()).hexdigest(), hashlib.md5(tissue.tobytes()).hexdigest()}

            for crop_img in raw_crops:
                h = hashlib.md5(crop_img.tobytes()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_crops.append(crop_img)

            # --- CHANGED: Save directly as separate datasets (No Stacking) ---
            try:
                g = hf.create_group(group_key)
                g.attrs['RegistrationCode'] = patient_id
                g.attrs['research_stage'] = visit_id
                g.attrs['NumCrops'] = len(unique_crops)

                g.create_dataset('bone', data=bone, compression='gzip')
                g.create_dataset('tissue', data=tissue, compression='gzip')

                if len(unique_crops) > 0:
                    c_grp = g.create_group('crops')
                    for i, c_dat in enumerate(unique_crops):
                        c_grp.create_dataset(str(i), data=c_dat, compression='gzip')

                count += 1
            except Exception: pass

    print(f"Done. Processed {count} subjects into {hdf5_path}")

if __name__ == "__main__":
    build_dataset()