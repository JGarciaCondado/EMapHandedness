from tensorflow.keras.models import load_model
from generator import Generator
import numpy as np
import xmippLib

class Predictor():
    """Predictor class used to generate predictions for a given model and
    evaluate the accuracy of the model.

    Arguments:
    model: model used to predict classes
    boxDim: box dimension used in model

    """

    def __init__(self, model, boxDim):
        # TODO get boxDim from model
        self.model = model
        self.boxDim = boxDim
        self.generator = Generator('', None, None, boxDim, None)

    def predict_volume_class(self, volume, mask):
        """Predict volume class from volume and mask return 
        both the overall class and each prediction.
        """
        
        # Empty generator boxes
        self.generator.Boxes = []

        # Populate boxes
        #TODO make mask optional
        self.generator.createBoxes(volume, mask)
        X = np.array(self.generator.Boxes)
        X = X.reshape(len(X), X.shape[1], X.shape[2], X.shape[3], 1)
        box_predictions = self.predict_box_class(X)
        volume_prediction = self.consensus_voting(box_predictions)
        return volume_prediction, box_predictions

    def predict_box_class(self, boxes):
        """Predict box class and return prediction.
        """
        return self.model.predict(boxes)

    def predict_pdb_class(self, pdb, maxRes, flip=False):
        """Predict pdb class from pdb file by simulating
        electorn density and passing it through the model.
        """

        # Simulate PDB to obtain volume and mask
        self.generator.maxRes = maxRes
        # TODO fix use hash function inside simulate volume
        V, Vmask = self.generator.simulate_volume(pdb, 'simpdb')
       
        #TODO apply flip

        #TODO what if PDB simulation unssucesfull

        #Obtain predictions
        Vpred, boxpred = self.predict_volume_class(V, Vmask)

        return Vpred, boxpred

    def evaluate_accuracy(self, targets, volumes, masks, threshold=0.5):
        """ Evaluate accuacy of model from targest, volumes and masks.
        """
        #TODO check length targets, volumes and masks same
        #TODO make masks optional
        predictions = []
        for volume, mask in zip(volumes, masks):
            volume_prediction, box_predictions = self.predict_volume_class(volume, mask)
            predictions.append(volume_prediction)
        predictions = np.array(predictions)
        return self.evaluate_accuracy_from_predictions(targets, predictions, threshold)
            
    def evaluate_accuracy_from_predictions(self, targets, predictions, threshold = 0.5):
        """ Evaluate the accuarcy from targets and provided predictiosn.
        """
        return np.sum((predictions >= threshold) == targets)/targets.size

    def evaluate_accuracy_generator(self, generator):
        """ Evaluate the accuracy of the generator by boxes for an epoch 
        """

        targets, predictions = [], []
        for i in range(len(generator)):
            X, Y = generator[i]
            pred = self.model.predict(X)
            targets.append(Y)
            predictions.append(pred)

        targets, predictions = np.array(targets).flatten(), np.array(predictions).flatten() 
        return self.evaluate_accuracy_from_predictions(targets, predictions)

       
    # TODO evaluate accuracy_generator PDB and boxes seperately

    def consensus_voting(self, predictions, threshold=0.5):
        """ Return a class given predictions by thresholding average. 
        """
        if np.sum(predictions)/predictions.size > 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    predictions = np.array([0.65, 0.87, 0.1, 0.6, 0.24, 0.8])
    targets = np.array([1, 0, 0, 1, 0, 0])
    print(predictor.consensus_voting(predictions))
    print(predictor.evaluate_accuracy_from_predictions(targets, predictions))
    volume = xmippLib.Image('test_files/test_vol.vol').getData()
    mask = xmippLib.Image('test_files/test_mask.vol').getData()
    volume_class, boxes_class = predictor.predict_volume_class(volume, mask)
    print(volume_class, boxes_class[:10])
    print(predictor.evaluate_accuracy(np.ones(1), [volume], [mask]))
    vpred, boxpred = predictor.predict_pdb_class('nrPDB/1EHS.pdb', 1)
    print(vpred, boxpred[:10])
    print(predictor.evaluate_accuracy_generator(Generator('nrPDB', 100, 100, 15, 1))) 
