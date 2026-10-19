import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window

# Resolve project paths relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def compute_ndwi_s2(raster):
    green = raster[2].astype(np.float32)
    nir = raster[7].astype(np.float32)
    denom = green + nir
    return np.where(denom != 0, (green - nir) / denom, 0.0)

def normalize_sar(band, bottom=-30.0, top=0.0):
    clipped = np.clip(band, bottom, top)
    return (clipped - bottom) / (top - bottom)

def preprocess_and_tile():
    out_img_dir = os.path.join(OUT_DIR, "images")
    out_mask_dir = os.path.join(OUT_DIR, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    s1_files = sorted(glob.glob(os.path.join(RAW_DIR, "s1", "*_S1Hand.tif")))
    print(f"Found {len(s1_files)} scene(s). Processing...")

    saved_patches = 0
    patch_size = 256
    stride = 256

    for s1_path in s1_files:
        base_name = os.path.basename(s1_path).replace("_S1Hand.tif", "")
        s2_path = os.path.join(RAW_DIR, "s2", f"{base_name}_S2Hand.tif")
        mask_path = os.path.join(RAW_DIR, "masks", f"{base_name}_LabelHand.tif")

        if not os.path.exists(s2_path) or not os.path.exists(mask_path):
            print(f"Skipping {base_name}: missing matching pair.")
            continue

        with rasterio.open(s1_path) as src_s1, \
             rasterio.open(s2_path) as src_s2, \
             rasterio.open(mask_path) as src_mask:

            rows, cols = src_mask.height, src_mask.width

            for r in range(0, rows - patch_size + 1, stride):
                for c in range(0, cols - patch_size + 1, stride):
                    window = Window(col_off=c, row_off=r, width=patch_size, height=patch_size)

                    # 1. Read ground truth mask slice
                    mask_slice = src_mask.read(1, window=window)
                    final_mask = np.where(mask_slice == 1, 1, 0).astype(np.uint8)

                    # 2. Read satellite sensor slices
                    s1_win = src_s1.read(window=window).astype(np.float32)
                    s2_win = src_s2.read(window=window).astype(np.float32)

                    # 3. Normalize channels to [0, 1]
                    ch_vv = normalize_sar(s1_win[0])
                    ch_vh = normalize_sar(s1_win[1])
                    ch_s2_nir = np.clip(s2_win[7] / 10000.0, 0.0, 1.0)
                    raw_ndwi = compute_ndwi_s2(s2_win)
                    ch_ndwi = np.clip((raw_ndwi + 1.0) / 2.0, 0.0, 1.0)

                    # 4. Stack into 4-channel image tensor: (4, 256, 256)
                    fused_patch = np.stack([ch_vv, ch_vh, ch_s2_nir, ch_ndwi], axis=0).astype(np.float32)
                    patch_transform = rasterio.windows.transform(window, src_s1.transform)
                    patch_name = f"{base_name}_x{c}_y{r}.tif"

                    # 5. Save 4-channel image (float32, nodata removed)
                    img_meta = src_s1.meta.copy()
                    img_meta.update({
                        "driver": "GTiff",
                        "count": 4,
                        "height": patch_size,
                        "width": patch_size,
                        "dtype": "float32",
                        "transform": patch_transform,
                        "nodata": None
                    })
                    with rasterio.open(os.path.join(out_img_dir, patch_name), "w", **img_meta) as dst:
                        dst.write(fused_patch)

                    # 6. Save single-channel mask (uint8, nodata removed)
                    mask_meta = src_mask.meta.copy()
                    mask_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": patch_size,
                        "width": patch_size,
                        "dtype": "uint8",
                        "transform": patch_transform,
                        "nodata": None
                    })
                    with rasterio.open(os.path.join(out_mask_dir, patch_name), "w", **mask_meta) as dst:
                        dst.write(final_mask, 1)

                    saved_patches += 1

    print(f"Done! Successfully generated {saved_patches} patches in '{OUT_DIR}'.")

if __name__ == "__main__":
    preprocess_and_tile()