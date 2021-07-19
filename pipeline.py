import torch
import numpy as np

from models import AlphaVolNet, EM3DNet_extended

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

box = torch.from_numpy(np.load(box_f))

model_conv = AlphaVolNet(trained_model)
print(model_conv.forward(box[None, None, :, :, :]))

model_FC = EM3DNet_extended(epochs=epochs, verbose=verbose, num_batches=num_batches_eval, save_folder=save_folder, restore=restore, init_model=init_model)
print(model_FC.forward(box[None, None, :, :, :]))
