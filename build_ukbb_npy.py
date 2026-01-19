import os
import zipfile
import io
import pydicom
import numpy as np
from scipy.ndimage import sobel
from tqdm import tqdm
import argparse
import cv2
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_DIR = '/net/mraid20/export/jasmine/UKBB_DATA/dexa_images'
OUTPUT_DIR = '/net/mraid20/export/jasmine/UKBB_DATA/preprocessed_dexa_images'

# Define separate folders for organization
FULLBODY_DIR = os.path.join(OUTPUT_DIR, 'fullbody')
CROPS_DIR = os.path.join(OUTPUT_DIR, 'crops')

# Aspect Ratio Thresholds
AR_FULLBODY_MIN = 2.3
AR_FULLBODY_MAX = 4.0  # Sanity check

# --- HELPER FUNCTIONS ---

def fix_photometry_and_normalize(ds):
    """
    Reads DICOM pixel array, handles physics slopes, auto-inverts white backgrounds,
    and performs SAFE normalization (avoids exploding noise).
    """
    # 1. Get Physical Values
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    arr = ds.pixel_array.astype(np.float32) * slope + intercept

    # 2. AUTO-INVERT: Corner Brightness Rule
    h, w = arr.shape
    if h < 20 or w < 20: return arr

    corners = np.concatenate([
        arr[0:10, 0:10].flatten(),
        arr[0:10, w-10:w].flatten(),
        arr[h-10:h, 0:10].flatten(),
        arr[h-10:h, w-10:w].flatten()
    ])

    # If corners are bright (closer to max than min), invert.
    if np.mean(corners) > (np.max(arr) + np.min(arr)) / 2:
        arr = np.max(arr) - arr

    # 3. Background Cleaning
    hist, bin_edges = np.histogram(arr, bins=100)
    mode_val = bin_edges[np.argmax(hist)]
    if mode_val < (np.max(arr) * 0.2):
        arr = arr - mode_val
        arr[arr < 0] = 0

    # 4. SAFE NORMALIZATION (The Fix)
    max_val = np.percentile(arr, 99.9)
    min_val = arr.min()
    current_range = max_val - min_val

    arr = np.clip(arr, min_val, max_val)

    # CHECK: Is this image actually contrasty?
    # Your data is 8-bit (0-255). If the range is small (e.g., < 40),
    # it's likely just empty tissue or noise. Don't stretch it.
    if current_range > 40:
        # High Dynamic Range (Bone exists) -> Stretch to 0.0 - 1.0
        if current_range > 1e-5:
            arr = (arr - min_val) / current_range
        else:
            arr = np.zeros_like(arr)
    else:
        # Low Dynamic Range (Empty/Faint) -> Keep it faint!
        # Scale against a standard "Bone Max" (255) instead of its own tiny max.
        arr = (arr - min_val) / 255.0

    return arr.astype(np.float32)

def get_sharpness(img):
    """Returns gradient magnitude sum. High = Bone, Low = Tissue."""
    sx = sobel(img, axis=0)
    sy = sobel(img, axis=1)
    return np.hypot(sx, sy).sum()

def process_zip(zip_path):
    patient_id = os.path.basename(zip_path).split('_')[0]
    visit_id = os.path.basename(zip_path).split('_')[2]

    # Dictionary to group candidates by their exact shape
    # Key: (rows, cols), Value: List of dictionaries
    shape_groups = defaultdict(list)
    crop_candidates = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for f in z.namelist():
                if not f.endswith('.dcm'): continue

                with z.open(f) as dcm_f:
                    try:
                        ds = pydicom.dcmread(io.BytesIO(dcm_f.read()), force=True)
                        rows = getattr(ds, 'Rows', 0)
                        cols = getattr(ds, 'Columns', 0)

                        if rows < 50 or cols < 50: continue

                        aspect_ratio = rows / cols
                        img_arr = fix_photometry_and_normalize(ds)

                        # --- LOGIC UPDATE: GROUPING ---
                        if (AR_FULLBODY_MIN < aspect_ratio < AR_FULLBODY_MAX) and (rows < 1000):
                            # It is a Full Body Scan -> Add to specific shape group
                            shape_groups[(rows, cols)].append({
                                'img': img_arr,
                                'sharpness': get_sharpness(img_arr),
                                'filename': f
                            })
                        else:
                            # It is a Crop
                            crop_candidates.append({
                                'img': img_arr,
                                'filename': f
                            })

                    except Exception as e:
                        print(f"Failed to process DICOM {f} in {zip_path}: {e}")
                        continue

        # --- FIND THE BEST MATCHING PAIR ---
        selected_pair = None

        # 1. Sort groups by resolution (Area = H * W) descending
        # We want the High Res pair if it exists
        sorted_shapes = sorted(shape_groups.keys(), key=lambda s: s[0]*s[1], reverse=True)

        for shape in sorted_shapes:
            candidates = shape_groups[shape]
            # We strictly need at least 2 images of the SAME size to form a pair
            if len(candidates) >= 2:
                selected_pair = candidates[:2] # Take the first 2 matching files
                break

        # If no valid pair found (e.g. only 1 high res and 1 low res), skip fullbody
        if selected_pair:
            img_a = selected_pair[0]
            img_b = selected_pair[1]

            # Bone/Tissue Logic (Same as before)
            if img_a['sharpness'] > img_b['sharpness']:
                bone = img_a['img']
                tissue = img_b['img']
            else:
                bone = img_b['img']
                tissue = img_a['img']

            composite = (bone + tissue) / 2.0

            final_tensor = np.stack([bone, tissue, composite], axis=0)

            # 4. Save as uint8
            final_tensor_uint8 = (final_tensor * 255).astype(np.uint8)
            np.save(os.path.join(FULLBODY_DIR, f"{patient_id}_{visit_id}_fullbody.npy"), final_tensor_uint8)

        # --- PROCESS CROPS ---
        for i, crop in enumerate(crop_candidates):
            # Save each crop as uint8 numpy array
            crop['img'] = (crop['img'] * 255).astype(np.uint8)
            save_name = os.path.join(CROPS_DIR, f"{patient_id}_{visit_id}_crop_{i}.npy")
            np.save(save_name, crop['img'])

        return True

    except Exception as e:
        print(f"Failed to process zip {zip_path}: {e}")
        return False

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(FULLBODY_DIR): os.makedirs(FULLBODY_DIR)
    if not os.path.exists(CROPS_DIR): os.makedirs(CROPS_DIR)

    zips = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.zip')]
    print(f"Found {len(zips)} zip files.")

    count = 0
    for z in tqdm(zips):
        if process_zip(z):
            count += 1

    print(f"Done. Processed {count} subjects.")