
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

    def predict_volume_class(self, volume, mask):
        """Predict volume class from volume and mask return 
        both the overall class and each prediction.
        """
        
        Zdim, Ydim, Xdim = volume.shape
        boxDim2 = self.boxDim//2
        X=[]
        for z in range(boxDim2,Zdim-boxDim2):
            for y in range(boxDim2,Ydim-boxDim2):
                for x in range(boxDim2,Xdim-boxDim2):
                    if mask[z,y,x]>0:
                        box = volume[z-boxDim2:z+boxDim2+1,y-boxDim2:y+boxDim2+1,x-boxDim2:x+boxDim2+1]
                        X.append(box/np.linalg.norm(box))
        X = np.array(X)#.astype('float32')
        X = X.reshape(len(X), X.shape[1], X.shape[2], X.shape[3], 1)
        #TODO make mask optional
        box_predictions = self.predict_box_class(X)
        volume_prediction = self.consensus_voting(box_predictions)
        return volume_prediciton, box_predictions

    def predict_box_class(self, boxes):
        """Predict box class and return prediction.
        """
        return self.model.predict(boxes)

    def predict_pdb_class(self, pdb, flip=False):
        """Predict pdb class from pdb file by simulating
        electorn density and passing it through the model.
        """
        pass

    def evaluat_accuarcy(self, targets, volumes, masks, threshold=0.5):
        """ Evaluate accuacy of model from targest, volumes and masks.
        """
        #TODO check length targets, volumes and masks same
        #TODO make masks optional
        predictions = []
        for volume, mask in zip(targets, volumes, masks):
            volume_prediction, box_predictions = self.predict_volume_class(volume, mask)
            predictions.append(volume_prediction)
        return self.evaluate_accuracy_from_predictions(targets, predictions, threshold)
            
    def evaluate_accuracy_from_predictions(self, targets, predictions, threshold = 0.5):
        """ Evaluate the accuarcy from targets and provided predictiosn.
        """
        return np.sum((predictions >= threshold) == targets)/targets.size

    def consensus_voting(self, predictions, threshold=0.5):
        """ 
        """
        if np.sum(predictions)/predictions.size > 0.5:
            return 1
        else:
            return 0
