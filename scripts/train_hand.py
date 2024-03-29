"""Train EM3DNet model from a dataset containing labels of hand."""

import torch
import os

from torch.utils.data import DataLoader
from hapi.data import HandDataset
from hapi.models import EM3DNet_extended

# Training variables
torchDataset_root = 'nrPDB/torchDataset/5ABeta/handDataset'
save_folder = 'Models/5ABeta/handedness'
restore = True
init_model = 'models/1A_hand_beta.pth'
batch_size = 1048
epochs = 50
verbose = 1
num_batches_eval = 1

# Load Dataset
traindataset = torch.load(os.path.join(torchDataset_root, 'trainDataset'))
valdataset = torch.load(os.path.join(torchDataset_root, 'valDataset'))
testdataset = torch.load(os.path.join(torchDataset_root, 'testDataset'))

# Create Loaders
trainloader = DataLoader(traindataset, batch_size=batch_size,
                         shuffle=True, num_workers=2)
valloader = DataLoader(valdataset, batch_size=batch_size,
                       shuffle=True, num_workers=2)
testloader = DataLoader(testdataset, batch_size=batch_size,
                        shuffle=True, num_workers=2)

# Initialize model and train model
model = EM3DNet_extended(epochs=epochs, verbose=verbose, num_batches=num_batches_eval,
                         save_folder=save_folder, restore=restore, init_model=init_model)
model.trainloop(trainloader, valloader)

# Evaluate performance of final model
print("Evaluating model at last epoch")
print("Train accuracy: %f" % model.eval_performance(
    trainloader, num_batches=len(trainloader)))
print("Validation accuracy: %f" % model.eval_performance(
    valloader, num_batches=len(valloader)))
print("Test accuracy: %f" % model.eval_performance(
    testloader, num_batches=len(testloader)))
