import opendatasets as od
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import torchvision


class ArtDataset(pl.LightningDataModule):
    def __init__(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError


class ArtGenerator(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ArtDiscriminator(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


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

# Download the data if not downloaded
od.download("https://www.kaggle.com/datasets/ansonnnnn/historic-art")
# Training loop
