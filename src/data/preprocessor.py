import glob
import numpy as np
import rasterio
import os
from rasterio.windows import Window


#computes the NDWI fronm Sentinel 2
def compute_ndwi_s2(raster):
    green = raster[2].astype(np.float32)
    nir = raster[7].astype(np.float32)

    ndwi = np.where(green + nir != 0 , (green - nir) / (green + nir),0.0)
    return ndwi