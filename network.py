from tensorflow.keras import layers, models, losses

class EMHNet():
    def __init__(self, boxDim):
        print('Build EMHNet...')
        self.boxDim = boxDim

    def create_model(self): 
        inputLayer = layers.Input(shape=(self.boxDim,self.boxDim,self.boxDim,1), name="input")
        L = layers.Conv3D(512, (15,15,15)) (inputLayer)
        L = layers.BatchNormalization()(L)
        L = layers.Flatten() (L)
        L = layers.Dense(1,name="output", activation="linear") (L)

        model = models.Model(inputLayer, L)
        model.compile(optimizer = 'adam',
                      loss = losses.BinaryCrossentropy(),
                      metric=['accuracy'])
        return model
