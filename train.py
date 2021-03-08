from torch.utils.data import DataLoader
from alpha_dataset import AlphaDataset
import torch
import os

if __name__ == '__main__':
    # Variables
    torchDataset_root = 'nrPDB/torchDataset'
    batch_size = 64
    # Load Dataset
    traindataset = torch.load(os.path.join(torchDataset_root, 'trainDataset'))
    valdataset = torch.load(os.path.join(torchDataset_root, 'valDataset'))
    # Create Loaders
    trainloader = DataLoader(traindataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    valloader = DataLoader(valdataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
