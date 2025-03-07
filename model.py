import opendatasets as od
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep
import os
import torchvision
import numpy as np


class ArtDataset(pl.LightningDataModule):
    def __init__(self):
        # Define transform
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512,512)),
            torchvision.transforms.ToTensor()
            ])
        path = os.path.join(".", "historic-art", "complete", "artwork")
        self.imgs = [os.path.join(path,i) for i in os.listdir(path)]

    def __getitem__(self, idx):
        print(self.imgs[idx])
        pil_img = Image.open(self.imgs[idx])
        tensor = self.transforms(pil_img)
        return tensor

    def __len__(self):
        return len(self.imgs)


class ArtDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Definition of the convolutional architecture
        self.model = torch.nn.Sequential(
                # First conv layer
                torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                # Second conv layer
                torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                # Third conv layer
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                # Linear layers
                torch.nn.Flatten(0),
                torch.nn.Linear(254016, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512,256),
                torch.nn.ReLU(),
                torch.nn.Linear(256,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                )

    def forward(self, x):
        try:
            logits = self.model(x)
            return logits
        except:
            return torch.zeros(1)


class ArtGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 256
        self.model = torch.nn.Sequential(
            # Input: latent vector z
            torch.nn.Linear(self.latent_size, 512 * 16 * 16),  
            torch.nn.Unflatten(-1, (512, 16, 16)),  # [batch, 512, 16, 16]
            
            # ConvTranspose Block 1
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256), 
            torch.nn.ReLU(True),
            
            # ConvTranspose Block 2 (256x32x32)
            #torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            #torch.nn.BatchNorm2d(128),
            #torch.nn.ReLU(True),
            
            # ConvTranspose Block 3 (128x64x64)
            #torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            #torch.nn.BatchNorm2d(64),
            #torch.nn.ReLU(True),
            
            # Final upsampling to 512x512
            #torch.nn.Upsample(scale_factor=8, mode='bilinear'),
            #torch.nn.Conv2d(64, 3, 3, padding=1),
            #torch.nn.Tanh()  # Output in [-1, 1]
        )


    def forward(self, x):
        tensor = torch.randn(self.latent_size)
        print(tensor.shape)
        logits = self.model(tensor)
        print(logits.shape)
        return logits


class ArtGAN(pl.LightningModule):
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self, x):
        raise NotImplementedError

    def training_step(self, idx, batch_idx):
        raise NotImplementedError

    def validation_step(self, idx, batch_idx):
        raise NotImplementedError


class ArtDataModule(pl.LightningModule):
    def __init__(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError



# Download the data if not downloaded
od.download("https://www.kaggle.com/datasets/ansonnnnn/historic-art")
# Training loop


# Debugging loop
dataset = ArtDataset()
#data = dataset[1234]
#discriminator = ArtDiscriminator()
#enc_state = discriminator(data)
#print(enc_state.shape)
gen = ArtGenerator()
xd = torch.randn(256)
gen(xd)






















