from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from network import EMHNet
from generator import Generator
import pickle

DEBUG = False
batch_size = 100
batch_per_epoch = 20
boxDim = 15
maxRes = 1


if __name__=="__main__":

    if DEBUG:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)

    with open('nrPDB/metadata/data_split.pkl', 'rb') as f:
        train_f, val_f, test_f = pickle.load(f)
    generator_train = Generator(train_f, batch_size, batch_per_epoch, boxDim, maxRes)
    generator_val = Generator(val_f, batch_size, batch_per_epoch, boxDim, maxRes)
    model = EMHNet(boxDim).create_model()
    model.summary()
    validationX, validationY = generator_val[0]
    checkpoint = ModelCheckpoint('detectHand_%1.1f.h5'%maxRes, monitor='loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir="logs", histogram_freq=1, write_graph=False, write_images=False)
    model.fit(generator_train,  epochs=30, verbose=1,
                        callbacks=[checkpoint, tensorboard],
                        validation_data=(validationX, validationY))
    model.save('detectHand_%1.1f.h5'%maxRes)
