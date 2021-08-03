import numpy as np
import os
import pandas as pd

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
thr = 0.75
batch_size = 2048

# Evaluate data
data_dir = 'nrPDB/Dataset/Volumes'
save_file = 'metrics/boxvolumes.csv'
PDB_names = []
precision_vals = []
TPs = []
FPs = []
predict_hands = []
handedness_vals = []
handedness_acc = []

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
    PDB_names.append(PDB[:-4])

    # Find alpha precision
    TP, FP, precision = pipeline.model_alpha.precision(Vf, Vmask, Vmask_SSE, thr, batch_size)
    precision_vals.append(precision)
    TPs.append(TP)
    FPs.append(FP)

    # Find handness
    pred_h, handedness = pipeline.evaluate(Vf, Vmask, thr, batch_size, hand)
    handedness_acc.append(handedness)
    handedness_vals.append(hand)
    predict_hands.append(pred_h)

pd.set_option('display.max_rows', None)
df = pd.DataFrame(list(zip(TPs, FPs, precision_vals, predict_hands, handedness_vals, handedness_acc)), index=PDB_names, columns=['TPs', 'FPs', 'Precision', 'Prediction', 'Label', 'Handness accuracy'])
df.to_csv(save_file)
print(df)
print("Average precision: %f" % np.nanmean(precision_vals))
print("Accuracy of handedness: %f" % np.nanmean(handedness_acc))
