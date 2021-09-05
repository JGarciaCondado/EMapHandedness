from hapi.simulator import create_SSE_dataset_exp
import mrcfile
import os
import numpy as np
import pandas as pd
import re

# Get pdb lists
pdbs = []
for PDB in os.listdir('nrPDB/PDB_EMDB_boxes/'):
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
pdb_files = ['nrPDB/PDB_EMDB_boxes/%s.pdb' % pdb.upper() for pdb in pdbs]
em_files = ['/home/jgarcia/data/emdb_agosto21/EMD-%s/map/emd_%s.map.gz' % (e_id, e_id) for e_id in emdb_ids]

# Dataset variables
dataset_root = 'nrPDB/Dataset/Exp_boxes'
minresidues = 7
maxRes = 5
mask_threshold = 0.5
box_dim = 11
SE_centroids = np.ones((3, 3, 3))
SE_noSSEMask = np.ones((3, 3, 3))

#Generate dataset
create_SSE_dataset_exp(pdb_files, em_files, dataset_root, maxRes,
                       mask_threshold, minresidues, box_dim,
                       SE_centroids, SE_noSSEMask, restart=False)
