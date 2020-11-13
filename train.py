from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from network import EMHNet
from generator import Generator


fnDir = 'nrPDB'
batch_size = 100
batch_per_epoch = 1
boxDim = 15
maxRes = 1


if __name__=="__main__":
    generator = Generator(fnDir, batch_size, batch_per_epoch, boxDim, maxRes) 
    model = EMHNet(boxDim).create_model()
    model.summary()
    validationX, validationY = generator[0]
    checkpoint = ModelCheckpoint('detectHand_%1.1f.h5'%maxRes, monitor='loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir="logs", histogram_freq=1, write_graph=False, write_images=False)
    model.fit(generator,  epochs=1, verbose=1,
                        callbacks=[checkpoint, tensorboard],
                        validation_data=(validationX, validationY))
    model.save('detectHand_%1.1f.h5'%maxRes)
