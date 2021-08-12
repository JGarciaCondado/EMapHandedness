"""Test pipeline on simulated volumes."""

import torch
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from hapi.models import HaPi
from hapi.data import VolDataset

# Load pipeline
alpha_model = '../Models/5A_alpha_model.pth'
hand_model = '../Models/5A_hand_model.pth'
box_dim = 11
c = 5
pipeline = HaPi(alpha_model, hand_model, box_dim, c)

# Predictions variables
thr = 0.5
batch_size = 2048

# Load data
torchDataset_root = '../nrPDB/torchDataset/Volumes'
dataset = torch.load(os.path.join(torchDataset_root, 'valDataset'))

# Evaluate data
save_file = '../metrics/boxvolumes.csv'
PDB_names = []
precision_vals = []
TPs = []
FPs = []
predict_hands = []
handedness_vals = []
handedness_acc = []

for PDB, hand, vol in tqdm(dataset):

    # Load data
    [Vf, Vmask, Vmask_SSE] = vol

    # Randomly flip volume
    if hand:
        axis = np.random.randint(3)
        Vf = np.flip(Vf, axis=axis)
        Vmask = np.flip(Vmask, axis=axis)
        Vmask_SSE = np.flip(Vmask_SSE, axis=axis)

    # Store name of PDB
    PDB_names.append(PDB)

    # Find alpha precision
    TP, FP, precision = pipeline.model_alpha.precision(
        Vf, Vmask, Vmask_SSE, thr, batch_size)
    precision_vals.append(precision)
    TPs.append(TP)
    FPs.append(FP)

    # Find handness
    pred_h, handedness = pipeline.evaluate(Vf, Vmask, thr, batch_size, hand)
    handedness_acc.append(handedness)
    handedness_vals.append(hand)
    predict_hands.append(pred_h)

pd.set_option('display.max_rows', None)
df = pd.DataFrame(list(zip(TPs, FPs, precision_vals,
                           predict_hands, handedness_vals, handedness_acc)),
                  index=PDB_names, columns=['TPs', 'FPs', 'Precision',
                                            'Prediction', 'Label',
                                            'Handness accuracy'])
df.to_csv(save_file)
print(df)
print("Average precision: %f" % np.nanmean(precision_vals))
print("Accuracy of handedness: %f" % np.nanmean(handedness_acc))
