"""Test pipeline on experimental data from EMDB."""

import numpy as np
import mrcfile

from hapi.models import HaPi
from hapi.processing import process_experimental_map
from scipy.ndimage.morphology import binary_erosion

# Load pipeline
alpha_model = 'Models/exp_alpha.pth'
hand_model = 'models/5A_hand_model.pth'
batch_size = 2048
pipeline = HaPi(alpha_model, hand_model)

# Load data
EMmap = 'nrPDB/test_exp/7rh5.map'
contour_level = 1.1 # From EMDB entry of authors
filter_res = 5.0 # Models trained at that resolution
Vf, Vmask = process_experimental_map(EMmap, filter_res, contour_level)

# Uncomment to flip
Vf = np.flip(Vf, axis=0)
Vmask = np.flip(Vmask, axis=0)

# Predict handedness
thr = 0.7
handness = pipeline.predict(Vf, Vmask, thr, batch_size)
print("Predicted handedness: %f" % handness)
