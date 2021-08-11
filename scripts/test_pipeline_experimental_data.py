import numpy as np
import mrcfile

from hapi.models import HaPi
from hapi.processing import process_experimental_map

# Load pipeline
alpha_model = '../Models/5A_alpha_model.pth'
hand_model = '../Models/5A_hand_model.pth'
box_dim = 11
c = 5
pipeline = HaPi(alpha_model, hand_model, box_dim, c)

# Load data
ID = 22882
EMmap = '../nrPDB/Exp_maps/maps/emd_%d.map' % ID
filter_res = 5.0
Vf, Vmask = process_experimental_map(EMmap, filter_res)

# Predict alphas
batch_size = 2048
alpha_probs = pipeline.model_alpha.predict_volume(Vf, Vmask, batch_size)
with mrcfile.new('../nrPDB/Exp_maps/alpha_maps/%s.map' % ID, overwrite=True) as mrc:
    mrc.set_data(np.zeros(alpha_probs.shape, dtype=np.float32))
    mrc.data[:, :, :] = alpha_probs

# Predict handedness
thr = 0.5
handness = pipeline.predict(Vf, Vmask, thr, batch_size)
print("Predicted handedness: %f" % handness)
