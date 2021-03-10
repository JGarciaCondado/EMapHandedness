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
    model = AlphaNet_extended(epochs=50, verbose=1, num_batches=1)
    model.trainloop(trainloader, valloader)

    # Evaluate performance
    print("Train accuracy: %f" % model.eval_performance(trainloader, num_batches=len(trainloader)))
    print("Validation accuracy: %f" % model.eval_performance(valloader, num_batches=len(valloader)))
    print("Test accuracy: %f" % model.eval_performance(testloader, num_batches=len(testloader)))

    # Load different model
    model = AlphaNet_extended(restore=True, file_name='25model_checkpoint.pth')

    # Evaluate performance
    print("Train accuracy: %f" % model.eval_performance(trainloader, num_batches=len(trainloader)))
    print("Validation accuracy: %f" % model.eval_performance(valloader, num_batches=len(valloader)))
    print("Test accuracy: %f" % model.eval_performance(testloader, num_batches=len(testloader)))
