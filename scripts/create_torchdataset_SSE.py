"""Creates torchdataset for training SSE from previously generated dataset."""

import torch
import os

from hapi.data import SSEDataset

# Torchdataset variables
dataset_root = 'nrPDB/Dataset/1ABeta'
torchDataset_root = 'nrPDB/torchDataset/1ABeta/betaDataset'
SSE_type = 'beta'
trainsplit, valsplit, testsplit = 0.7, 0.15, 0.15
flip = True
noise = None

# Generate Dataset
dataset = SSEDataset(dataset_root, SSE_type, flip, noise)

# Split into different Datasets
trainsize = int(len(dataset)*trainsplit)
valsize = int(len(dataset)*valsplit)
testsize = len(dataset)-trainsize-valsize
trainDataset, valDataset, testDataset = torch.utils.data.random_split(
    dataset, [trainsize, valsize, testsize])

# Save Datasets
torch.save(trainDataset, os.path.join(torchDataset_root, 'trainDataset'))
torch.save(valDataset, os.path.join(torchDataset_root, 'valDataset'))
torch.save(testDataset, os.path.join(torchDataset_root, 'testDataset'))
