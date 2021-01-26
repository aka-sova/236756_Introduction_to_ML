import pandas as pd  # data analysis and manipulation tool
import numpy as np  # Numerical computing tools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K #for additional calculations

from sklearn.preprocessing import OneHotEncoder #one-hot encode 

class transfer_learn():
    def __init__(self, X_train, X_test, y_train, y_test,X_bin_train, X_bin_test, y_bin_train, y_bin_test ):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.X_bin_train=X_bin_train
        self.X_bin_test=X_bin_test
        self.y_bin_train=y_bin_train
        self.y_bin_test=y_bin_test
    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
        
    def weight_training(self):

        #define MLP model
        model = keras.Sequential(
            [
            layers.Dense(11, activation="relu", name="layer_1",input_shape=(11,)),
            layers.Dense(150, activation="relu", name="layer_2"),
            layers.Dense(100, activation="relu", name="layer_3"),
            layers.Dense(100, activation="relu", name="layer_4"),
            layers.Dense(100, activation="relu", name="layer_5"),
            layers.Dense(50, activation="relu", name="layer_6"),
            layers.Dense(1, name="layer_7")
            ]
        )

        model.summary() # print model layout


        #copmile model
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(), self.recall_m], #add-on of recall not required
        )
        #train model
        epochs = 200
        model.fit(x=np.array(self.X_bin_train),y=np.array(self.y_bin_train), epochs=epochs, validation_data=(np.array(self.X_bin_test), np.array(self.y_bin_test)))


        #save weights for later re-use, change path location ?
        model.save_weights('my_weights.h5', overwrite=True)
    
    def trans_learn(self):
    #define model that fits weights
        model = keras.Sequential(
        [
        layers.Dense(11, activation="relu", name="layer_1",input_shape=(11,)),
        layers.Dense(150, activation="relu", name="layer_2"),
        layers.Dense(100, activation="relu", name="layer_3"),
        layers.Dense(100, activation="relu", name="layer_4"),
        layers.Dense(100, activation="relu", name="layer_5"),
        layers.Dense(50, activation="relu", name="layer_6"),
        layers.Dense(1, name="layer_7")
        ]
        )



        # Presumably you would want to first load pre-trained weights.
        model.load_weights('my_weights.h5')


        #copy to another model
        t_model = keras.Sequential()
        for layer in model.layers[:-4]: # exclude some last layers
            t_model.add(layer)
        for layer in t_model.layers:
            layer.trainable = False #backpropegation only on new layers
        #add new trainable layers
        t_model.add(layers.Dense(100, activation="relu", name="layer_8"))
        t_model.add(layers.Dense(100, activation="relu", name="layer_20"))
        t_model.add(layers.Dense(10, activation="relu", name="layer_9"))
        t_model.add(layers.Dense(6, activation='softmax',name="layer_10"))

        t_model.summary()

        

        #oen hot encode y_test for multiclass classification
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.array(self.y_train).reshape(-1,1))
        y_train_enc = enc.transform(np.array(self.y_train).reshape(-1,1)).toarray()
        y_test_enc = enc.transform(np.array(self.y_test).reshape(-1,1)).toarray()


        # Recompile and train (this will only update the weights of non-first layers.
        t_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy(), self.recall_m],
        )

        epochs = 250
        t_model.fit(x=np.array(self.X_train),y=np.array(y_train_enc), epochs=epochs, validation_data=(np.array(self.X_test), np.array(y_test_enc)))

        #t_model.predict(X) for prediction


