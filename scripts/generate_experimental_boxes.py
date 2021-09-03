from hapi.simulator import create_SSE_dataset_exp
import mrcfile
import os
import numpy as np


if __name__ == "__main__":
    pdb_files = ['7nhr.pdb', '7rh5.pdb', '7oqy.pdb']
    em_files = ['7nhr.map', '7rh5.map', '7oqy.map']
    dataset_root = 'Exp_boxes'
    minresidues = 7
    maxRes = 5
    mask_threshold = 0.5
    box_dim = 11
    SE_centroids = np.ones((3, 3, 3))
    SE_noSSEMask = np.ones((3, 3, 3))
    create_SSE_dataset_exp(pdb_files, em_files, dataset_root, maxRes,
                           mask_threshold, minresidues, box_dim,
                           SE_centroids, SE_noSSEMask, restart=False)
