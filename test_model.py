import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchdataset_hand import HandDataset
from torchdataset_SSE import SSEDataset
from models import EM3DNet_extended

if __name__ == '__main__':

    # Variables
    torchDataset_root = 'nrPDB/torchDataset/beta/SSE'
    model_directory = 'Models/Beta/SSE'
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

    # Load best model
    model = EM3DNet_extended(restore=True, save_folder=model_directory, init_model=epoch+'model_checkpoint.pth')

    # Evaluate performance
    print("Evaluate model at epoch: %s" % epoch)
    print("Train accuracy: %f" % model.eval_performance(trainloader, num_batches=len(trainloader)))
    print("Validation accuracy: %f" % model.eval_performance(valloader, num_batches=len(valloader)))
    print("Test accuracy: %f" % model.eval_performance(testloader, num_batches=len(testloader)))
