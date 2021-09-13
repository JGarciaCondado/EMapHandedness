"""Test pipeline on simulated volumes."""

import os
import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from hapi.models import HaPi
from hapi.processing import process_experimental_map

# Load pipeline
alpha_model = 'models/5A_SSE_experimental.pth'
hand_model = 'models/5A_TL_hand_alpha.pth'
pipeline = HaPi(alpha_model, hand_model)

# Predictions variables
thr = 0.7
filter_res = 5.0
batch_size = 2048

# Load data
df_emdb = pd.read_csv('data/EMDB_metadata.csv')
IDs = [int(ID[4:]) for ID in df_emdb['emdb_id'].tolist()]
print(IDs)
map_f = 'nrPDB/Exp_hand_dataset/EMD-%d/map/emd_%d.map.gz'
header_f = 'nrPDB/Exp_hand_dataset/EMD-%d/header/emd-%d-v30.xml'

# Evaluate data
save_file = 'metrics/emdb_hands.csv'
hand_vals = []
ID_finished = []

for ID in tqdm(IDs):

    # Define files
    EMmap = map_f % (ID, ID)
    header = header_f % (ID, ID)

    # Check files exist
    if os.path.isfile(EMmap) and os.path.isfile(header):

        # Find contour level from emdb entry
        contour_level = float(ET.parse(header).getroot().findall('.//level')[0].text)

        # Process data
        Vf, Vmask = process_experimental_map(EMmap, filter_res, contour_level)

        # Find handness
        handness = pipeline.predict(Vf, Vmask, thr, batch_size)

        # Print if high handness
        if handness > 0.6:
            print(ID, handness)

    else:
        handness = None

    #Append values
    hand_vals.append(handness)
    ID_finished.append(ID)

    #Save csv at each iteration
    df = pd.DataFrame(hand_vals, index=ID_finished, columns=['Hand'])
    df.to_csv(save_file)

print(df)
