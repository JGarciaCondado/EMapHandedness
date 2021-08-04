import numpy as np
import os
import pandas as pd
import torch

from dataset_generator import simulate_volume, create_directory
from tqdm import tqdm
from torch.utils.data import Dataset

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
        if os.path.isfile(dataset_root+'/%s.npy'%PDB[:-4]) and not restart:
            continue
        #Obtain volumes
        Vf, Vmask, Vmask_SSE = simulate_volume(data_root+PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
        if Vf is not None and Vmask is not None and Vmask_SSE is not None:
            # Save all volumes 
            data = np.stack([Vf, Vmask, Vmask_SSE])
            np.save(dataset_root+'/%s.npy'%PDB[:-4], data)

class VolDataset(Dataset):
    """ Torch Dataset object that loads the Volumes previously created
        assinging them a label of handedness randomly.
    """
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.dataset_table = None
        self._init_dataset()

    def __len__(self):
        return len(self.dataset_table.index)

    def __getitem__(self, idx):
        pdb, label = self.dataset_table.iloc[idx, :]
        pdb_id = os.path.join(self.dataset_root, pdb)
        vol = np.load(pdb_id)
        return (pdb[:-4], label, vol)

    def _init_dataset(self):
        """ Create pandas DataFrame to store all the PDBs available
        """
        dataset_info = []
        for PDB in os.listdir(self.dataset_root):
            # Randomly assing label flipped or not flipped
            label = np.random.choice(2)
            dataset_info.append([PDB, label])
        self.dataset_table = pd.DataFrame(dataset_info, columns=['PDB', 'Label'])

if __name__ == "__main__":

    # Define variables
    data_root = 'nrPDB/PDB_test/'
    dataset_root = 'nrPDB/Dataset/Volumes'
    torchDataset_root = 'nrPDB/torchDataset/Volumes'
    SSE_type = 'alpha'
    maxRes = 5.0
    mask_threshold = 0.5
    SSE_mask_threshold = 0.5
    minresidues = 4
    restart = False
    valsplit = 1.0

    # Create dataset
    create_volume_dataset(data_root, dataset_root, maxRes, mask_threshold, SSE_mask_threshold,SSE_type, minresidues, restart)

    # Generate Dataset object
    dataset = VolDataset(dataset_root)
    # Split into different Datasets
    valsize = int(len(dataset)*valsplit)
    testsize = len(dataset)-valsize
    valDataset, testDataset = torch.utils.data.random_split(dataset, [valsize, testsize])
    # Save Datasets
    torch.save(valDataset, os.path.join(torchDataset_root, 'valDataset'))
    torch.save(testDataset, os.path.join(torchDataset_root, 'testDataset'))
