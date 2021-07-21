import torch
import numpy as np

from models import AlphaVolNet
from dataset_generator import simulate_volume

# Load model
model_state = 'Models/5A_alpha_model.pth'
box_dim = 11
c = 5
model = AlphaVolNet(model_state, box_dim, c)

# Load data
PDB = 'nrPDB/PDB/1AGC.pdb'
SSE_type = 'alpha'
maxRes = 5.0
mask_threshold = 0.5
SSE_mask_threshold = 0.5
minresidues = 4

Vf, Vmask, Vmask_SSE = simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)

# Predict
batch_size = 2048
alpha_probs = model.predict_volume(Vf, Vmask, batch_size)
np.save('nrPDB/Examples/alphas/estimated_alphas.npy', alpha_probs)
