import glob
import numpy as np
import rasterio
import os
from rasterio.windows import Window


#computes the NDWI fronm Sentinel 2(Normal difference water index )
def compute_ndwi_s2(raster):
    green = raster[2].astype(np.float32)
    nir = raster[7].astype(np.float32)

    ndwi = np.where(green + nir != 0 , (green - nir) / (green + nir),0.0)
    return ndwi

def preprocess_and_tile(raw_d = "data/raw",out_d="data/processed",patch_size = 256,stride = 256):
    
    #creates the ouput folder if it does not exist yet
    outIm_d = os.path.join(out_d,"images")
    out_mask_d = os.path.join(out_d,"masks")
    os.makedirs(outIm_d,exist_ok=True)
    os.makedirs(out_mask_d,exist_ok=True)

    
    s1_files = glob.glob(os.path.join(raw_d,"s1","_S1Hand.tif"))
    sorted_s1_files = sorted(s1_files)

    cout_patch_saved = 0

    for s1_path in s1_files:
        base_name = os.path.basename(s1_path).replace("_S1Hand.tif")
        s2_path = os.path.join(raw_d,"s2",f"{base_name}_S2Hand.tif")
        mask_path = os.path.join(raw_d,"masks",f"{base_name}_LabelHand.tif")

        #Opens al 3 files with rasterio
        with rasterio.open(s1_path) as src_s1, rasterio.open(s2_path) as src_s2,rasterio.open(mask_path) as src_mask:
            rows = src_mask.height
            cols = src_mask.width

            for r in range(0,rows - patch_size + 1,stride):
                for c in range(0,cols - patch_size +1 ,stride):
                    window = Window(col_off = c,row_off = r,width = patch_size,height = patch_size)
                    mask_slice = src_mask.read(1,window = window)
                    final_mask = np.where(mask_slice == 1,1,0).astype(np.uint8)

                #read raw data
                s1_win = src_s1.read(window=window).astype(np.float32)
                s2_win = src_s2.read(window=window).astype(np.float32)

                def normalize_sar(band,bottom=-30.0,top=0.0):
                    clipped = np.clip(band,bottom,top)
                    return (clipped - bottom) / (top - bottom)

                #Normalized channels to [0,1]
                ch_vv = normalize_sar(s1_win[0])
                ch_vh = normalize_sar(s1_win[1])
                ch_s2_nir = np.clip(s2_win[7] / 1000.0,0.0,1.0)

                #Water index shift from [0,1]
                raw_ndwi = compute_ndwi_s2(s2_win)
                ch_ndwi = np.clip((raw_ndwi + 1.0)/2.0 , 0.0 , 1.0)

                #stack all the channels together
                fused_patch = np.stack([ch_vv,ch_vh,ch_s2_nir,ch_ndwi],axis=0)
                patch_transform = rasterio.windows.transform(window,src_s1.transform)
                h_name = f"{base_name}_x{x}_y{y}.tif"

                



                

            

