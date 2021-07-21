import numpy as np

from models import HaPi
from dataset_generator import simulate_volume

# Load pipeline
alpha_model = 'Models/5A_alpha_model.pth'
hand_model = 'Models/5A_hand_model.pth'
box_dim = 11
c = 5
pipeline = HaPi(alpha_model, hand_model, box_dim, c)

# Load data
PDB = 'nrPDB/PDB/1ADG.pdb'
SSE_type = 'alpha'
maxRes = 5.0
mask_threshold = 0.5
SSE_mask_threshold = 0.5
minresidues = 4

Vf, Vmask, Vmask_SSE = simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
Vf = np.flip(Vf, axis=0)

# Predict hand
thr = 0.8
batch_size = 2048
handness = pipeline.predict(Vf, Vmask, thr, batch_size)
print(handness)
