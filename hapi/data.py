"""Dataset pytorch clases for HaPi package."""

import os
import numpy as np
import pandas as pd
import torch


class SSEDataset(torch.utils.data.Dataset):
    """Loads boxes previously created and assigns them SSE labels."""
    def __init__(self, dataset_root, SSE_type, c, flip):
        self.dataset_root = dataset_root
        self.dataset_table = None
        self.SSE_type = SSE_type
        self.c = c
        self.flip = flip
        self._init_dataset()

    def __len__(self):
        return len(self.dataset_table.index)

    def __getitem__(self, idx):
        pdb, box_type, box_n, label = self.dataset_table.iloc[idx, :]
        box_id = os.path.join(self.dataset_root, pdb,
                              box_type, 'box%s.npy' % box_n)
        box = self.transform(np.load(box_id))
        # If argument flip true randomly flip 50% of the boxes
        if self.flip and np.random.randint(2):
            box = np.flip(box, axis=np.random.randint(3)).copy()
        box = torch.from_numpy(box)
        # Add extra dimension as torch expects a channel dimension
        box = box.unsqueeze(0)
        label = torch.tensor(float(label))
        return (label, box)

    def _init_dataset(self):
        """ Create pandas DataFrame to store all the infromation of all the
            available boxes so that later they can be loaded when needed
            instead of loading all arrays.
        """
        dataset_info = []
        for PDB in os.listdir(self.dataset_root):
            pdb_folder = os.path.join(self.dataset_root, PDB)
            for box_type in os.listdir(pdb_folder):
                boxes_folder = os.path.join(pdb_folder, box_type)
                if box_type == self.SSE_type:
                    label = 1
                else:
                    label = 0
                for box in os.listdir(boxes_folder):
                    dataset_info.append([PDB, box_type, box[3:-4], label])
        self.dataset_table = pd.DataFrame(
            dataset_info, columns=['PDB', 'Box Type', 'Number', 'Label'])

    def transform(self, box):
        """Normalizes boxes to the range 0 to 1."""
        # Change negative values to zeros
        box[box < 0.0] = 0.0
        # Change values greater than c to c
        box[box > self.c] = self.c
        # Normalize to [0,1]
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box


class HandDataset(torch.utils.data.Dataset):
    """Loads boxes previously created and assigns them a hand."""

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
        box_id = os.path.join(self.dataset_root, pdb,
                              box_type, 'box%s.npy' % box_n)
        box = self.transform(np.load(box_id))
        if label:
            box = np.flip(box, axis=np.random.randint(3)).copy()
        box = torch.from_numpy(box)
        # Add extra dimension as torch expects a channel dimension
        box = box.unsqueeze(0)
        label = torch.tensor(float(label))
        return (label, box)

    def _init_dataset(self):
        """ Create pandas DataFrame to store all the infromation of all the available
            boxes so that later they can be loaded when needed instead of loading
            all arrays.
        """
        dataset_info = []
        for PDB in os.listdir(self.dataset_root):
            pdb_folder = os.path.join(self.dataset_root, PDB)
            boxes_folder = os.path.join(pdb_folder, self.SSE_type)
            for box in os.listdir(boxes_folder):
                # Randomly assing label flipped or not flipped
                label = np.random.choice(2)
                dataset_info.append([PDB, self.SSE_type, box[3:-4], label])
        self.dataset_table = pd.DataFrame(
            dataset_info, columns=['PDB', 'Box Type', 'Number', 'Label'])

    def transform(self, box):
        """Normalizes boxes to the range 0 to 1."""
        # Change negative values to zeros
        box[box < 0.0] = 0.0
        # Change values greater than c to c
        box[box > self.c] = self.c
        # Normalize to [0,1]
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box


class VolDataset(torch.utils.data.Dataset):
    """Loads volumes assinging them randomly a hand label."""

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
        self.dataset_table = pd.DataFrame(
            dataset_info, columns=['PDB', 'Label'])
