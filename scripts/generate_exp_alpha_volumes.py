import os
import torch
import pandas as pd
import re

from hapi.simulator import create_volume_dataset_exp
from hapi.data import VolDataset

# Get pdb lists
pdbs = []
for PDB in os.listdir('nrPDB/PDB_EMDB_test/'):
    if PDB[-4:] != '.pdb':
        continue
    else:
        pdbs.append(PDB[:-4].lower())

# Get emdb metadata csv
df_emdb = pd.read_csv('data/EMDB_metadata.csv')

# Obtain list of emdbs from pdbs
df_emdb_pdbs = df_emdb.loc[df_emdb['fitted_pdbs'].isin(pdbs)]
df_emdb_pdbs = df_emdb_pdbs.drop_duplicates(subset=['fitted_pdbs'])
emdbs = df_emdb_pdbs['emdb_id'].tolist()
emdb_ids = [re.findall(r'\d+', emdb_id)[0] for emdb_id in emdbs]
pdbs = df_emdb_pdbs['fitted_pdbs'].tolist()

# Create appropriate root files
pdb_f = ['nrPDB/PDB_EMDB_test/%s.pdb' % pdb.upper() for pdb in pdbs]
emdb_f = ['/home/jgarcia/data/emdb_agosto21/EMD-%s/map/emd_%s.map.gz' % (e_id, e_id) for e_id in emdb_ids]

# Dataset variables
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
dataset = VolDataset(dataset_root)

# Save Datasets
torch.save(dataset, os.path.join(torchDataset_root, 'testDataset'))
