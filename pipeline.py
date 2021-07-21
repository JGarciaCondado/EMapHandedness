import torch
import numpy as np

from models import AlphaVolNet, HandNet
from dataset_generator import simulate_volume

# Load alpha model
model_state = 'Models/5A_alpha_model.pth'
box_dim = 11
c = 5
model_alpha = AlphaVolNet(model_state, box_dim, c)

# Load hand net
model_state = 'Models/5A_hand_model.pth'
model_hand = HandNet(model_state, box_dim, c)

# Load data
PDB = 'nrPDB/PDB/1ADG.pdb'
SSE_type = 'alpha'
maxRes = 5.0
mask_threshold = 0.5
SSE_mask_threshold = 0.5
minresidues = 4

Vf, Vmask, Vmask_SSE = simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
#Vf = np.flip(Vf, axis=0)

# Predict hand
batch_size = 2048
alpha_probs = model_alpha.predict_volume(Vf, Vmask, batch_size)

# Alpha mask thresholding
alpha_mask = alpha_probs > 0.8

# Predict hand
handness = model_hand.predict_volume_consensus(Vf, alpha_mask, batch_size)
print(handness)
