from torch.utils.data import DataLoader
from alpha_dataset import AlphaDataset
import torch
import os

if __name__ == '__main__':
    # Variables
    dataset_root = 'nrPDB/Dataset'
    torchDataset_root = 'nrPDB/torchDataset'
    # Load Dataset
    dataset = torch.load(os.path.join(torchDataset_root, 'Dataset'))
    # Loader
    dataloader = DataLoader(dataset, batch_size=64)
    for batch in dataloader:
        print(batch[0])
