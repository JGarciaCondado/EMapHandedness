import torch
import time
import numpy as np

from torch import nn
from torch import optim

class AlphaNet(nn.Module):
    """ 3D CNN to estiamte the probability that there is an alpha helix in the center of the box.
    """
    def __init__(self):
        super().__init__()

        # Convlutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4,
                               kernel_size = 5, padding = 1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8,
                               kernel_size = 5, padding = 1)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16,
                               kernel_size = 3, padding = 0)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size = 3, padding = 0)
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size = 2, padding = 0)

        # Linear layers
        self.linear1 = nn.Linear(512,128)
        self.linear2 = nn.Linear(128,1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through the CNN operations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        # Flatten the tensor into a vecto
        x = x.view(-1,512)
        # Pass the tensor through the FC layes
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class AlphaNet_extended(AlphaNet):

    def __init__(self,epochs=100,lr=0.001, verbose=1):

        super().__init__()

        self.lr = lr #Learning Rate

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.epochs = epochs

        self.verbose = verbose

        self.criterion = nn.BCELoss()

        # A list to store the loss evolution along training

        self.loss_during_training = []

        self.valid_loss_during_training = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def trainloop(self,trainloader,validloader):

        # Optimization Loop

        for e in range(int(self.epochs)):

            start_time = time.time()

            # Random data permutation at each epoch

            running_loss = 0.

            for labels, images in trainloader:

                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                #Reset Gradients!
                self.optim.zero_grad()

                out = self.forward(images)

                loss = self.criterion(out.squeeze(),labels)

                running_loss += loss.item()

                #Compute gradients
                loss.backward()

                #SGD stem
                self.optim.step()


            self.loss_during_training.append(running_loss/len(trainloader))

            # Validation Loss

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                running_loss = 0.

                for labels, images in validloader:

                    # Move input and label tensors to the default device
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Compute output for input minibatch
                    out = self.forward(images)

                    #Your code here
                    loss = self.criterion(out.squeeze(),labels)

                    running_loss += loss.item()

                self.valid_loss_during_training.append(running_loss/len(validloader))


            if(e % self.verbose == 0):

                print("Epoch %d. Training loss: %f, Validation loss: %f, Time per epoch: %f seconds"
                      %(e,self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))

    def eval_performance(self,dataloader,num_batches=10,threshold=0.5):

        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():

            it_images = iter(dataloader)

            for e in range(int(num_batches)):
                labels, images = next(it_images)
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)
                probs = self.forward(images)
                pred_labels = probs.squeeze().numpy()>=threshold
                n_correct = np.sum(pred_labels == labels.numpy().astype('bool'))
                accuracy += n_correct/len(labels)

        return accuracy/int(num_batches)
