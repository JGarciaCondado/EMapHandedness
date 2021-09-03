"""Creates torchdataset for training hand from previously generated dataset."""

import os
import torch

from hapi.data import HandDataset

# Torchdataset variables
dataset_root = 'nrPDB/Dataset/1ABeta/'
torchDataset_root = 'nrPDB/torchDataset/1ABeta/handDataset'
SSE_type = 'beta'
trainsplit, valsplit, testsplit = 0.7, 0.15, 0.15

# Generate Dataset
dataset = HandDataset(dataset_root, SSE_type)

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
