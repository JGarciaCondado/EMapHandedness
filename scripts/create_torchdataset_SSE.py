import sys
sys.path.insert(0, "../")

import torch
import os

from hapi.data import SSEDataset

# Variables
dataset_root = '../nrPDB/Dataset/5A'
torchDataset_root = '../nrPDB/torchDataset/alphaDataset'
SSE_type = 'alpha'
trainsplit, valsplit, testsplit = 0.7, 0.15, 0.15
c = 5
flip = True

# Generate Dataset
dataset = SSEDataset(dataset_root, SSE_type, c, flip)

# Split into different Datasets
trainsize = int(len(dataset)*trainsplit)
valsize = int(len(dataset)*valsplit)
testsize = len(dataset)-trainsize-valsize
trainDataset, valDataset, testDataset = torch.utils.data.random_split(dataset, [trainsize, valsize, testsize])

# Save Datasets
torch.save(trainDataset, os.path.join(torchDataset_root, 'trainDataset'))
torch.save(valDataset, os.path.join(torchDataset_root, 'valDataset'))
torch.save(testDataset, os.path.join(torchDataset_root, 'testDataset'))
