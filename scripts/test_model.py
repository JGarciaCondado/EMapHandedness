"""Test a trained EM3DNet model with given dataset to find best epoch."""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from hapi.data import SSEDataset, HandDataset
from hapi.models import EM3DNet_extended

# Variables
torchDataset_root = 'nrPDB/torchDataset/alphaDataset'
model_directory = 'models/5A_SSE_alpha.pth'
batch_size = 1024

# Load validation and test loss
trainloss = np.load(os.path.join(model_directory, 'trainloss.npy'))
valloss = np.load(os.path.join(model_directory, 'valloss.npy'))
plt.plot(range(len(trainloss)), trainloss, label='training')
plt.plot(range(len(valloss)), valloss, label='validation')
plt.legend()
plt.show()

# Ask for epoch
epoch = input('Epoch to evaluate at: ')

# Load Test Dataset
testdataset = torch.load(os.path.join(torchDataset_root, 'testDataset'))
testloader = DataLoader(testdataset, batch_size=batch_size,
                        shuffle=True, num_workers=2)

# Load best model
model = EM3DNet_extended(restore=True, save_folder=model_directory,
                         init_model=model_directory+epoch+'model_checkpoint.pth')

# Evaluate performance
print("Evaluate model at epoch: %s" % epoch)
print("Test accuracy: %f" % model.eval_performance(
    testloader, num_batches=len(testloader)))

# Evaluate precision
print("Test precision: %f" % model.eval_precision(
    testloader, num_batches=len(testloader)))
