import os
import torch

from hapi.simulator import create_volume_dataset_exp
from hapi.data import VolDataset

# Variables
pdb_f = ['nrPDB/test_exp/7rh5.pdb']
emdb_f = ['nrPDB/test_exp/7rh5.map']
dataset_root = 'nrPDB/Dataset/Experimental_Volumes'
torchDataset_root = 'nrPDB/torchDataset/Experimental_Volumes'
maxRes = 5.0
mask_threshold = 0.5
minresidues = 7
restart = False

# Create dataset
create_volume_dataset_exp(pdb_f, emdb_f, dataset_root, maxRes, mask_threshold,
                          minresidues, restart)

# Generate Dataset object
dataset = VolDataset(torchDataset_root)

# Save Datasets
torch.save(dataset, os.path.join(torchDataset_root, 'testDataset'))
