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

    sample_im = im_files[0]
    file_name = os.path.basename(sample_im)
    sample_mask_path = os.path.join("data/processed/masks",file_name)
    with rasterio.open(sample_im) as src1, rasterio.open(sample_mask_path) as src2:

        image = src1.read().astype(np.float32)
        ground_truth = src2.read(1).astype(np.uint8)

    tensor_input = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        proba = torch.sigmoid(logits).cpu().numpy()
        predicted_mask = (proba >0.5).astype(np.uint8)

    os.makedirs("reports/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    #SAR VV Radar backsatter
    axes[0].imshow(image[0],cmap="gray")
    axes[0].set_title("Sentinel-1 SAR VV")
    axes[0].axis("off")

    #NDWI
    axes[1].imshow(image[3],cmap="Blues")
    axes[1].set_title("NDWI")
    axes[1].axis("off")
    