import os
import glob
import rasterio
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


def evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = smp.Unet(encoder_name="resnet34",classes=1,in_channels=4,encoder_weights=4)

    model.load_state_dict(torch.load("models/flood_u.pth",map_location=device))
    model.to(device)
    model.eval()

    im_file_path = "data/processed/images/*.tif"
    im_files = sorted(glob.glob(im_file_path))
    