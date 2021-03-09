import torch
import os

from torch.utils.data import DataLoader
from alpha_dataset import AlphaDataset
from models import AlphaNet_extended

if __name__ == '__main__':
    # Variables
    torchDataset_root = 'nrPDB/torchDataset'
    batch_size = 64

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
    model = AlphaNet_extended(epochs=100, verbose=20)
    model.trainloop(trainloader, valloader)

    # Evaluate performance
    print("Train accuracy: %f" % model.eval_performance(trainloader, num_batches=2))
    print("Validation accuracy: %f" % model.eval_performance(valloader, num_batches=1))
    print("Test accuracy: %f" % model.eval_performance(testloader, num_batches=1))
