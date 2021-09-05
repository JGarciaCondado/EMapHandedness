import os
import torch

from hapi.simulator import create_volume_dataset_pdb
from hapi.data import VolDataset

# Variables
data_root = 'nrPDB/PDB_test/'
dataset_root = 'nrPDB/Dataset/Volumes'
torchDataset_root = 'nrPDB/torchDataset/Volumes'
SSE_type = 'alpha'
maxRes = 5.0
mask_threshold = 0.5
SSE_mask_threshold = 0.5
minresidues = 7
restart = False
valsplit = 0.3

# Create dataset
create_volume_dataset_pdb(data_root, dataset_root, maxRes, mask_threshold,
    SSE_mask_threshold, SSE_type, minresidues, restart)

# Generate Dataset object
dataset = VolDataset(torchDataset_root)

# Split into different Datasets
valsize = int(len(dataset)*valsplit)
testsize = len(dataset)-valsize
valDataset, testDataset = torch.utils.data.random_split(
    dataset, [valsize, testsize])

# Save Datasets
torch.save(valDataset, os.path.join(torchDataset_root, 'valDataset'))
torch.save(testDataset, os.path.join(torchDataset_root, 'testDataset'))
