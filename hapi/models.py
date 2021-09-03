"""Define models, train and set parameters.

Used in the HaPi package to declare models, train models on data and
to load trained models for execution.

Classes
-------
EMD3Net - 3D CNN to estimate labels from a set of boxes.
EMD3Net_extended - Wrapper of EMD3Net for training.
AlphaVolNet - Wrapper of EM3DNet to predict alpha helices in volumes.
HandNet - Wrapper of EM3DNet to predict handedness of a volume.
HaPi - Pipeline to determine the hanededness of a volume.

"""

import torch
import time
import numpy as np
import os

from torch import nn
from torch import optim


class EM3DNet(nn.Module):
    """ 3D CNN to estiamte labels from a set of boxes."""

    def __init__(self):
        super().__init__()

        # Convlutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4,
                               kernel_size=5, padding=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8,
                               kernel_size=5, padding=1)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16,
                               kernel_size=3, padding=0)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=0)
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=2, padding=0)

        # Linear layers
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assumes boxes have dimension 11x11x11
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
        # Flatten the tensor into a vector
        x = x.view(-1, 512)
        # Pass the tensor through the FC layes
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class EM3DNet_extended(EM3DNet):
    """ Wrapper for EM3DNet for training. """

    def __init__(self, epochs=100, lr=0.001, verbose=1, num_batches=10,
                 save_folder='Models', restore=False, save_e=1,
                 file_name='model_checkpoint.pth', init_model=None):

        super().__init__()

        self.lr = lr  # Learning Rate

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.epochs = epochs

        self.verbose = verbose

        self.num_batches = num_batches

        self.criterion = nn.BCELoss()

        self.save_folder = save_folder

        self.file_name = file_name

        self.init_model = init_model

        self.save_e = save_e

        if torch.cuda.is_available():
            print("GPU sucessfully found")
            self.device = torch.device("cuda")
        else:
            print("GPU NOT found, using cpu")
            self.device = torch.device("cpu")

        self.to(self.device)

        if restore:
            if self.init_model is not None:
                state_dict = torch.load(
                    os.path.join(self.init_model),
                    map_location=self.device)
            else:
                state_dict = torch.load(
                    os.path.join(self.save_folder, self.file_name),
                    map_location=self.device)
            self.load_state_dict(state_dict)

        # A list to store the loss evolution along training

        self.loss_during_training = []

        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):
        """ Train model on given dataset."""

        # Optimization Loop

        for e in range(int(self.epochs)):

            start_time = time.time()

            # Random data permutation at each epoch

            running_loss = 0.

            for labels, boxes in trainloader:

                # Move input and label tensors to the default device
                boxes, labels = boxes.to(self.device), labels.to(self.device)

                # Reset Gradients!
                self.optim.zero_grad()

                out = self.forward(boxes)

                loss = self.criterion(out.squeeze(), labels)

                running_loss += loss.item()

                # Compute gradients
                loss.backward()

                # SGD stem
                self.optim.step()

            self.loss_during_training.append(running_loss/len(trainloader))

            # Validation Loss

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                running_loss = 0.

                for labels, boxes in validloader:

                    # Move input and label tensors to the default device
                    boxes, labels = boxes.to(
                        self.device), labels.to(self.device)

                    # Compute output for input minibatch
                    out = self.forward(boxes)

                    # Your code here
                    loss = self.criterion(out.squeeze(), labels)

                    running_loss += loss.item()

                self.valid_loss_during_training.append(
                    running_loss/len(validloader))

            if(e % self.save_e == 0):
                torch.save(self.state_dict(), os.path.join(
                    self.save_folder, str(e)+self.file_name))
                np.save(os.path.join(self.save_folder, 'valloss'),
                        self.valid_loss_during_training)
                np.save(os.path.join(self.save_folder, 'trainloss'),
                        self.loss_during_training)
            if(e % self.verbose == 0):
                print("Epoch %d. Training loss: %f, Validation loss: %f, " \
                      "Train accuracy: %f, Validation accuracy: %f, " \
                      "Time per epoch: %f seconds"
                      % (e, self.loss_during_training[-1],
                         self.valid_loss_during_training[-1],
                         self.eval_performance(trainloader, self.num_batches),
                         self.eval_performance(validloader, self.num_batches),
                         (time.time() - start_time)), sep='')

    def eval_performance(self, dataloader, num_batches=10, threshold=0.5):
        """ Evaluate the model accuracy on a given dataset."""

        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():

            it_boxes = iter(dataloader)

            for e in range(int(num_batches)):
                labels, boxes = next(it_boxes)
                # Move input and label tensors to the default device
                boxes, labels = boxes.to(self.device), labels.to(self.device)
                probs = self.forward(boxes)
                pred_labels = probs.cpu().squeeze().numpy() >= threshold
                n_correct = np.sum(
                    pred_labels == labels.cpu().numpy().astype('bool'))
                accuracy += n_correct/len(labels)

        return accuracy/int(num_batches)


class AlphaVolNet(EM3DNet):
    """Model that detects alpha helices within a volume."""

    def __init__(self, trained_model):

        super().__init__()

        # Box dimensions set at 11x11x11 from design of CNN
        self.box_dim = 11

        # Load devices availbale
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

        # Load model
        self.model_file = trained_model
        state_dict = torch.load(self.model_file, map_location=self.device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(self, x):
        """Predict a label for each box."""

        # Load data to available device
        x = x.to(self.device)

        # Pass through network
        pred = self.forward(x)

        return pred

    def normalize(self, box):
        """ Normalize boxes to between 0 and 1."""

        box[box < 0] = 0
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box

    def predict_volume(self, Vf, Vmask, batch_size):
        """Predict regions within volume that contain alpha helices."""

        # Array to store probabilities
        alpha_probs = np.zeros(Vf.shape)

        # Get coordinates of mask
        coors = np.argwhere((Vmask))

        # Variables for boxes
        box_hw = int((self.box_dim-1)/2)
        n_batches = int(np.ceil(coors.shape[0]/batch_size))

        # Pack set of boxes into batches to pass through network
        for i in range(n_batches):

            # Last batch is smaller
            if i == n_batches-1:
                batch = coors.shape[0] % batch_size
            else:
                batch = batch_size

            # Array to store batches
            boxes = np.zeros([batch, self.box_dim, self.box_dim, self.box_dim])

            # Obtain for each batch the normalized box at each loaction
            for j in range(batch):
                x, y, z = coors[i*batch_size+j]
                box = Vf[x-box_hw:x+box_hw+1, y -
                         box_hw:y+box_hw+1, z-box_hw:z+box_hw+1]
                # If box not the correct shape ignore
                if box.shape == (self.box_dim, self.box_dim, self.box_dim):
                    boxes[j, :, :, :] = self.normalize(box)

            # Run through network
            boxes = torch.from_numpy(boxes.astype(np.float32))[
                :, None, :, :, :]
            predictions = self.predict(boxes)

            # Store in appropriate part of the volume
            for j, pred in enumerate(predictions.flatten()):
                x, y, z = coors[i*batch_size+j]
                alpha_probs[x, y, z] = float(pred)

        return alpha_probs

    def precision(self, Vf, Vmask, Vmask_alpha, thr, batch_size):
        """Given a threshold for probabilities evaluate precision."""

        # Obtain location of alphahelics
        alpha_probs = self.predict_volume(Vf, Vmask, batch_size)

        # Alpha mask by thresholding
        alpha_mask = alpha_probs > thr

        # Find number of true positivies and false positives
        TP = np.sum(np.logical_and(alpha_mask, Vmask_alpha))
        FP = np.sum(np.logical_and(alpha_mask, np.logical_not(Vmask_alpha)))

        # If no TP or FP return nan to avoid division by zero
        if TP+FP == 0:
            precision = np.nan
        else:
            precision = TP/(TP+FP)

        return TP, FP, precision


class HandNet(EM3DNet):
    """Model that predicts a volumes handedness from given alpha mask."""

    def __init__(self, trained_model):

        super().__init__()

        # Box dimensions set at 11x11x11 from design of CNN
        self.box_dim = 11

        # Load devices availbale
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

        # Load model
        self.model_file = trained_model
        state_dict = torch.load(self.model_file, map_location=self.device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(self, x):
        """Predict a label for each box."""

        # Load data to available device
        x = x.to(self.device)

        # Pass through network
        pred = self.forward(x)

        return pred

    def normalize(self, box):
        """Normalize boxes to between 0 and 1."""

        box[box < 0] = 0
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box

    def predict_volume_consensus(self, Vf, Alpha_mask, batch_size):
        """Predict volume handedness by means of consensus voting."""

        # Hand predictions
        hand_predictions = np.zeros(np.sum(Alpha_mask))

        # Get coordinates of alphas
        coors = np.argwhere((Alpha_mask))

        # Variables for boxes
        box_hw = int((self.box_dim-1)/2)
        n_batches = int(np.ceil(coors.shape[0]/batch_size))

        # Pack set of boxes into batches to pass through network
        for i in range(n_batches):

            # Last batch is smaller
            if i == n_batches-1:
                batch = coors.shape[0] % batch_size
            else:
                batch = batch_size

            # Array to store batches
            boxes = np.zeros([batch, self.box_dim, self.box_dim, self.box_dim])

            # Obtain for each batch the normalized box at each loaction
            for j in range(batch):
                x, y, z = coors[i*batch_size+j]
                box = Vf[x-box_hw:x+box_hw+1, y -
                         box_hw:y+box_hw+1, z-box_hw:z+box_hw+1]
                boxes[j, :, :, :] = self.normalize(box)

            # Run through network
            boxes = torch.from_numpy(boxes.astype(np.float32))[
                :, None, :, :, :]
            predictions = self.predict(boxes)

            # Store predictions
            hand_predictions[i*batch_size:i*batch_size +
                             batch] = predictions.cpu().flatten().numpy()

        # Consensus voting is by taking average
        consensus_predictions = np.mean(hand_predictions)

        return consensus_predictions


class HaPi():
    """ Ha(ndedness) Pi(peline) is a model that predicts from a given
        electron density map volume its handedness.
    """

    def __init__(self, alpha_model, hand_model):

        # Load models
        self.model_alpha = AlphaVolNet(alpha_model)
        self.model_hand = HandNet(hand_model)

    def predict(self, Vf, Vmask, thr, batch_size):
        """ Predict handedness of an electron desnity map.

        A label of 0 means that it has correct handedness, conversely
        a label of 1 means that it has incorrect hand.

        Parameters:
        -----------
        Vf -- Numpy array of volume density values
        Vmask -- Nummpy array mask of non-background values
        thr -- Threshold for the probabilities of alpha estimation
        batch_size -- Number of boxes to pass to prediction model at time

        """

        # Obtain location of alphahelics
        alpha_probs = self.model_alpha.predict_volume(Vf, Vmask, batch_size)

        # Alpha mask by thresholding
        alpha_mask = alpha_probs > thr

        # Predict hand
        if alpha_mask.any():
            handness = self.model_hand.predict_volume_consensus(
                Vf, alpha_mask, batch_size)
        else:
            handness = None

        return handness

    def evaluate(self, Vf, Vmask, thr, batch_size, label):
        """ Evaluate wether prediction matches given label."""

        # Predict hand
        prediction = self.predict(Vf, Vmask, thr, batch_size)

        if prediction is None:
            result = np.nan
        else:
            # Assign hand
            hand = prediction > 0.5

            # Check wether it is correct
            result = hand == label

        return prediction, result
