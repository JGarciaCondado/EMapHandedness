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
                      metrics=['accuracy'])
        return model

class CNN3D():
    # TODO creation of 3DCNN define kernel size etc.. in initzialization
    def __init__(self, boxDim):
        print('Build 3DCNN...')
        self.boxDim = boxDim

    def create_model(self):
        inputLayer = layers.Input(shape=(self.boxDim,self.boxDim,self.boxDim,1), name="input")
        L = layers.Conv3D(8, (7,7,7), strides=(2,2,2), activation='relu')(inputLayer)
        L = layers.Conv3D(16, (5,5,5), strides=(2,2,2), activation='relu')(L)
        L = layers.Conv3D(32, (3,3,3), strides=(2,2,2), activation='relu')(L)
        L = layers.Flatten()(L)
        L = layers.Dense(128, activation='relu') (L)
        out = layers.Dense(1,name="output", activation="sigmoid") (L)

        model = models.Model(inputLayer, out)
        model.compile(optimizer = 'adam',
                      loss = losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model
