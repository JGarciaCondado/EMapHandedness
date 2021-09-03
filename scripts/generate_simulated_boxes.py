"""Simulate boxes that contain SSE and no SSE from given PDBs."""

import numpy as np

from hapi.simulator import create_SSE_dataset_pdb

# Variables for simulation
data_root = 'nrPDB/PDB/'
dataset_root = 'nrPDB/Dataset/Beta/'
SSE_type = 'beta'
maxRes = 1.0
mask_threshold = 0.5
SSE_mask_threshold = 0.5
minresidues = 4
box_dim = 11
SE_centroids = np.ones((2, 2, 2))
SE_noSSEMask = np.ones((3, 3, 3))
restart = True

# Create dataset
create_SSE_dataset_pdb(data_root, dataset_root, maxRes, mask_threshold,
                       SSE_mask_threshold, SSE_type, minresidues, box_dim,
                       SE_centroids, SE_noSSEMask, restart)
