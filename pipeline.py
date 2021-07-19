import torch
import numpy as np

from models import AlphaVolNet, EM3DNet_extended
from dataset_generator import simulate_volume

# Required inputs
trained_model = 'Models/alpha_model.pth'
box_f = 'nrPDB/Dataset/5A/1AGC/no_alpha/box2.npy'
save_folder = 'Models/'
init_model = 'alpha_model.pth'
restore = True
batch_size = 1048
epochs = 50
verbose = 1
num_batches_eval= 1
PDB = 'nrPDB/PDB/1AGC.pdb'
SSE_type = 'beta'
maxRes = 5.0
mask_threshold = 0.5
SSE_mask_threshold = 0.5
minresidues = 4

box = torch.from_numpy(np.load(box_f))

model_conv = AlphaVolNet(trained_model)
print(model_conv.forward(box[None, None, :, :, :]))

model_FC = EM3DNet_extended(epochs=epochs, verbose=verbose, num_batches=num_batches_eval, save_folder=save_folder, restore=restore, init_model=init_model)
print(model_FC.forward(box[None, None, :, :, :]))


Vf, Vmask, Vmask_SSE = simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
Vf[Vf>5] = 5
Vf[Vf<0] = 0
Vf = (Vf-np.min(Vf))/(np.max(Vf)-np.min(Vf))
Vf = torch.from_numpy(Vf)
print(Vf.shape)
result = model_conv.forward(Vf[None, None,:,:,:])
print(result.shape)
np.save('nrPDB/Examples/alphas/estimated_alphas.npy', result.numpy())
