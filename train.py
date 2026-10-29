import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import segmentation_models_pytorch as smp


class FloodDataset(Dataset):
    def __init__(self,data_d="data/processed"):
        self.im_paths = sorted(glob.glob(os.path.join(data_d, "images", "*.tif")))
        self.mask_d = os.path.join(data_d, "masks")


    def __getitem__(self, index):
        im_path = self.im_paths[index]
        mask_path = os.path.join(self.mask_d,os.path.basename(self.im_paths))

        with rasterio.open(self.im_path) as src1 , rasterio.open(mask_path) as src2:
            image = src1.read().astype(np.float32)
            mask = src2.read(1).astype(np.float32)
        im_t = torch.from_numpy(image)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return im_t,mask_t

    def __len__(self):
        return len(self.im_paths)



def train():

    #Loading dataset 
    dataset = FloodDataset(data_d="data/processed")
    loader = DataLoader(dataset,batch_size=2,shuffle=True)

    #Compute
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    #Model
    model = smp.Unet(encoder_name="resnet34",encoder_weights=None,in_channels=4,classes=1).to(device)

    #Loss
    closs = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    epochs = 10

    model.train()
    for epoch in range(epoch):
        epoch_loss = 0.0
        for im ,mask in loader:
            images = im.to(device)
            masks = mask.to(device)

            out = model(images)
            optimizer.zero_grad
            loss = closs(out,masks)
            loss.backward()
            optimizer.step()

    