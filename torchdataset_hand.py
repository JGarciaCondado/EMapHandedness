import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class HandDataset(Dataset):
    """ Torch Dataset object that load the boxes previously created and flips them
    """
    def __init__(self, dataset_root, SSE_type, c):
        self.dataset_root = dataset_root
        self.dataset_table = None
        self.SSE_type = SSE_type
        self.c = c
        self._init_dataset()

    def __len__(self):
        return len(self.dataset_table.index)

    def __getitem__(self, idx):
        pdb, box_type, box_n, label = self.dataset_table.iloc[idx, :]
        box_id = os.path.join(self.dataset_root, pdb, box_type, 'box%s.npy'%box_n)
        box = self.transform(np.load(box_id))
        if label:
            box = np.flip(box, axis=0).copy()
        box = torch.from_numpy(box)
        # Add extra dimension as torch expects a channel dimension
        box = box.unsqueeze(0)
        label =  torch.tensor(float(label))
        return (label, box)

    def _init_dataset(self):
        """ Create pandas DataFrame to store all the infromation of all the available
            boxes so that later they can be loaded when needed instead of loading all arrays.
        """
        dataset_info = []
        for PDB in os.listdir(self.dataset_root):
            pdb_folder = os.path.join(dataset_root, PDB)
            boxes_folder = os.path.join(pdb_folder, self.SSE_type)
            for box in os.listdir(boxes_folder):
                # Randomly assing label flipped or not flipped
                label = np.random.choice(2)
                dataset_info.append([PDB, self.SSE_type, box[3:-4], label])
        self.dataset_table = pd.DataFrame(dataset_info, columns=['PDB', 'Box Type', 'Number', 'Label'])

    def transform(self, box):
        """ Normalizes boxes to the range 0 to 1.
        """
        # Change negative values to zeros
        box[box<0.0] = 0.0
        # Change values greater than c to c
        box[box>self.c] = self.c
        # Normalize to [0,1]
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box


if __name__ == '__main__':
    # Variables
    dataset_root = 'nrPDB/Dataset/1A/'
    torchDataset_root = 'nrPDB/torchDataset/handDataset'
    SSE_type = 'alpha'
    trainsplit, valsplit, testsplit = 0.7, 0.15, 0.15
    c = 5
    # Generate Dataset
    dataset = HandDataset(dataset_root, SSE_type, c)
    # Split into different Datasets
    trainsize = int(len(dataset)*trainsplit)
    valsize = int(len(dataset)*valsplit)
    testsize = len(dataset)-trainsize-valsize
    trainDataset, valDataset, testDataset = torch.utils.data.random_split(dataset, [trainsize, valsize, testsize])
    # Save Datasets
    torch.save(trainDataset, os.path.join(torchDataset_root, 'trainDataset'))
    torch.save(valDataset, os.path.join(torchDataset_root, 'valDataset'))
    torch.save(testDataset, os.path.join(torchDataset_root, 'testDataset'))
