"""Test pipeline on experimental data from EMDB."""

import numpy as np
import xml.etree.ElementTree as ET

from hapi.models import HaPi
from hapi.processing import process_experimental_map

# Load pipeline
alpha_model = 'models/5A_SSE_experimental.pth'
hand_model = 'models/5A_TL_hand_alpha.pth'
batch_size = 2048
pipeline = HaPi(alpha_model, hand_model)

# Load data
ID = 12339
EMmap = 'nrPDB/Exp_hand_dataset/EMD-%d/map/emd_%d.map.gz' % (ID, ID)
header = 'nrPDB/Exp_hand_dataset/EMD-%d/header/emd-%d-v30.xml' % (ID, ID)
contour_level = float(ET.parse(header).getroot().findall('.//level')[0].text) # From EMDB entry of authors
filter_res = 5.0 # Models trained at that resolution
Vf, Vmask = process_experimental_map(EMmap, filter_res, contour_level)

# Uncomment to flip
#Vf = np.flip(Vf, axis=0)
#Vmask = np.flip(Vmask, axis=0)

# Predict handedness
thr = 0.7
handness = pipeline.predict(Vf, Vmask, thr, batch_size)
print("Predicted handedness: %f" % handness)
