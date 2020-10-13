
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
        #TODO make mask optional
        pass

    def predict_box_class(self, box):
        """Predict box class and return prediction.
        """
        #TODO make it work for multiple boxes
        pass

    def predict_pdb_class(self, pdb, flip=False):
        """Predict pdb class from pdb file by simulating
        electorn density and passing it through the model.
        """
        pass

    def evaluat_accuarcy(self, targets, volumes, masks):
        """ Evaluate accuacy of model from targest, volumes and masks.
        """
        #TODO check length targets, volumes and masks same
        #TODO make masks optional
        pass

    def evaluate_accuracy_from_predictions(self, targets, predictions):
        """ Evaluate the accuarcy from targets and provided predictiosn.
        """
        pass

    def consensus_voting(self, predictions):
        """ Return the class for which there are more predictions of such class.
        """
        pass
