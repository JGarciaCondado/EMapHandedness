import os
import numpy as np
import pandas as pd

dataset_root = 'nrPDB/Dataset'
dataset_info = []
for PDB in os.listdir(dataset_root):
    pdb_folder = os.path.join(dataset_root, PDB)
    for box_type in os.listdir(pdb_folder):
        boxes_folder = os.path.join(pdb_folder, box_type)
        for box in os.listdir(boxes_folder):
            dataset_info.append([PDB, box_type, box[3:-4]])
dataset_frame = pd.DataFrame(dataset_info, columns=['PDB', 'Box Type', 'Number'])
pdb, box_type, box_n = dataset_frame.iloc[30]
box_id = os.path.join(dataset_root, pdb, box_type, 'box%s.npy'%box_n)
print(box_id)
