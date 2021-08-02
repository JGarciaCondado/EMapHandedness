import numpy as np
import os

from models import HaPi
from dataset_generator import simulate_volume
from tqdm import tqdm

# Load pipeline
alpha_model = 'Models/5A_alpha_model.pth'
hand_model = 'Models/5A_hand_model.pth'
box_dim = 11
c = 5
pipeline = HaPi(alpha_model, hand_model, box_dim, c)

# Variable pipeline
thr = 0.8
batch_size = 2048

# Evaluate data
data_dir = 'nrPDB/Dataset/Volumes'
PDB_names = []
precision_vals = []
handedness_vals = []

for PDB in tqdm(os.listdir(data_dir)):

    # Load data
    [Vf, Vmask, Vmask_SSE] = np.load(os.path.join(data_dir, PDB))

    # Randomly flip volume
    if np.random.randint(2):
        Vf = np.flip(Vf, axis=np.random.randint(3))
        hand = 1
    else:
        hand = 0

    # Store name of PDB 
    PDB_names.append(PDB)

    # Find alpha precision
    precision = pipeline.model_alpha.precision(Vf, Vmask, Vmask_SSE, thr, batch_size)
    precision_vals.append(precision)

    # Find handness
    handedness = pipeline.evaluate(Vf, Vmask, thr, batch_size, hand)
    handedness_vals.append(handedness)

print(PDB_names, precision_vals, handedness_vals)

print("Average precision: %f" % np.nanmean(precision_vals))
print("Accuracy of handedness: %f" % np.nanmean(handedness_vals))
