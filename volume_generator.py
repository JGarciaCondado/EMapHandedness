import numpy as np
import os

from dataset_generator import simulate_volume, create_directory
from tqdm import tqdm

def create_volume_dataset(data_root, dataset_root, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues, restart=False):
    """ Creat the whole dataset from a directory containg all the PDB
    """
    # Create dataset direcotry if it doesn't exist
    create_directory(dataset_root)
    for PDB in tqdm(os.listdir(data_root)):
        # Ensure we are working with pdb files
        if PDB[-4:] != '.pdb':
            continue
        # If PDB dataset already there in case errors cause restart
        # Ignore if we want to redo the complete dataset
        if os.path.isdir(dataset_root+PDB[:-4]) and not restart:
            continue
        #Obtain volumes
        Vf, Vmask, Vmask_SSE = simulate_volume(data_root+PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
        if Vf is not None and Vmask is not None and Vmask_SSE is not None:
            # Save all volumes 
            data = np.stack([Vf, Vmask, Vmask_SSE])
            np.save(dataset_root+'/%s.npy'%PDB[:-4], data)

if __name__ == "__main__":
    # Define variables
    data_root = 'nrPDB/PDB/'
    dataset_root = 'nrPDB/Dataset/Volumes'
    SSE_type = 'alpha'
    maxRes = 1.0
    mask_threshold = 0.5
    SSE_mask_threshold = 0.5
    minresidues = 4
    restart = False

    # Create dataset
    create_volume_dataset(data_root, dataset_root, maxRes, mask_threshold, SSE_mask_threshold,SSE_type, minresidues, restart)
