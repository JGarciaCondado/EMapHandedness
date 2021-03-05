import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class AlphaDataset(Dataset):
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.dataset_table = None
        self._init_dataset()

    def __len__(self):
        return len(self.dataset_table.index)

    def __getitem__(self, idx):
        pdb, box_type, box_n, label = self.dataset_table.iloc[idx, :]
        box_id = os.path.join(dataset_root, pdb, box_type, 'box%s.npy'%box_n)
        box = torch.from_numpy(np.load(box_id))
        label =  torch.tensor(float(label))
        return (label, box)

    def _init_dataset(self):
        dataset_info = []
        for PDB in os.listdir(self.dataset_root):
            pdb_folder = os.path.join(dataset_root, PDB)
            for box_type in os.listdir(pdb_folder):
                boxes_folder = os.path.join(pdb_folder, box_type)
                if box_type == 'alpha':
                    label = 1
                else:
                    label = 0
                for box in os.listdir(boxes_folder):
                    dataset_info.append([PDB, box_type, box[3:-4], label])
        self.dataset_table = pd.DataFrame(dataset_info, columns=['PDB', 'Box Type', 'Number', 'Label'])

if __name__ == '__main__':
    # Variables
    dataset_root = 'nrPDB/Dataset'
    # Geenerate Dataset
    dataset = AlphaDataset(dataset_root)
    loader = DataLoader(dataset, batch_size=64,shuffle=True,num_workers=2)
    for i, batch in enumerate(loader):
        print(i, batch)
