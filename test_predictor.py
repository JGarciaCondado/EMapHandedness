"""Test the predict module."""
import pytest
import numpy as np
import xmippLib

from predict import Predictor
from generator import Generator
from tensorflow.keras.models import load_model

#TODO use fixtures and parametrize
#TODO add more examples

def test_consensus_voting(): 
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    predictions = np.array([0.65, 0.87, 0.1, 0.6, 0.24, 0.8])
    assert predictor.consensus_voting(predictions) == 1

def test_evaluate_accuaracy_from_predictions(): 
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    predictions = np.array([0.65, 0.87, 0.1, 0.6, 0.24, 0.8])
    targets = np.array([1, 0, 0, 1, 0, 0])
    assert predictor.evaluate_accuracy_from_predictions(targets, predictions) == 2/3

#TODO give especific example that should give certain values
def test_predict_volume_class(): 
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    volume = xmippLib.Image('test_files/test_vol.vol').getData()
    mask = xmippLib.Image('test_files/test_mask.vol').getData()
    volume_class, boxes_class = predictor.predict_volume_class(volume, mask)
    assert volume_class == 1 or volume_class == 0

def test_evaluate_accuracy(): 
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    volume = xmippLib.Image('test_files/test_vol.vol').getData()
    mask = xmippLib.Image('test_files/test_mask.vol').getData()
    volume_class, boxes_class = predictor.predict_volume_class(volume, mask)
    assert predictor.evaluate_accuracy(np.ones(1), [volume], [mask]) == 0.0

def test_predict_pdb_class():
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    vpred, boxpred = predictor.predict_pdb_class('nrPDB/1EHS.pdb', 1)
    assert vpred == 1 or vpred == 0

def test_evaluate_accuarcy_generator():
    model = load_model('model.h5')
    predictor = Predictor(model, 15)
    accuracy =  predictor.evaluate_accuracy_generator(Generator('nrPDB', 100, 100, 15, 1)) 
    assert accuracy > 0.0 and accuracy < 1.0
