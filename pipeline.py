import numpy as np
import mrcfile

from models import HaPi

# Load pipeline
alpha_model = 'Models/5A_alpha_model.pth'
hand_model = 'Models/5A_hand_model.pth'
box_dim = 11
c = 5
pipeline = HaPi(alpha_model, hand_model, box_dim, c)

# Load data
PDB = '6s1n'
EMmap = 'nrPDB/Exp_maps/maps/%s.map' % PDB
with mrcfile.open(EMmap) as mrc:
    Vf = mrc.data.copy()

# Threshold to create map
Vmask = np.zeros(Vf.shape)
thr_mask = 0.004
Vmask[Vf>thr_mask] = 1.0

# Predict alphas
batch_size = 2048
alpha_probs = pipeline.model_alpha.predict_volume(Vf, Vmask, batch_size)
with mrcfile.new('nrPDB/Exp_maps/alpha_maps/%s.map' % PDB, overwrite=True) as mrc:
    mrc.set_data(np.zeros(alpha_probs.shape, dtype=np.float32))
    mrc.data[:, :, :] = alpha_probs

# Predict handedness
thr = 0.5
handness = pipeline.predict(Vf, Vmask, thr, batch_size)
print("Predicted handedness: %f" % handness)
