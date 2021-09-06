"""Test pipeline on simulated volumes."""

import torch
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from hapi.models import AlphaVolNet
from hapi.data import VolDataset

# Load pipeline
alpha_model = 'models/5A_SSE_alpha.pth'
model = AlphaVolNet(alpha_model)

# Predictions variables
thr = 0.7
batch_size = 2048

# Load data
torchDataset_root = 'nrPDB/torchDataset/Experimental_Volumes'
dataset = torch.load(os.path.join(torchDataset_root, 'testDataset'))

# Evaluate data
save_file = 'metrics/experimental_precision_simulated_model.csv'
PDB_names = []
precision_vals = []
TPs = []
FPs = []

for PDB, hand, vol in tqdm(dataset):

    # Load data
    [Vf, Vmask, Vmask_SSE] = vol

    # Store name of PDB
    PDB_names.append(PDB)

    # Find alpha precision
    TP, FP, precision = model.precision(
        Vf, Vmask, Vmask_SSE, thr, batch_size)
    precision_vals.append(precision)
    TPs.append(TP)
    FPs.append(FP)

#pd.set_option('display.max_rows', None)
df = pd.DataFrame(list(zip(TPs, FPs, precision_vals)),
                  index=PDB_names, columns=['TPs', 'FPs', 'Precision'])
df.to_csv(save_file)
print(df)
print("Average precision: %f" % np.nanmean(precision_vals))
