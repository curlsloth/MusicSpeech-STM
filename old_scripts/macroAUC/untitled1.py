#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:50:28 2024

@author: andrewchang
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback

class MacroAUC(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        val_x, val_y = self.validation_data
        val_pred = self.model.predict(val_x)
        auc = self.macro_auc(val_y, val_pred)
        print(f"\nEpoch {epoch+1} - Macro AUC: {auc}")
        logs['val_macro_auc'] = auc

    def macro_auc(self, y_true, y_pred):
        # Assuming y_true is one-hot encoded
        n_classes = y_true.shape[1]
        aucs = []
        for i in range(n_classes):
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                pass
        return np.mean(aucs)


from keras_tuner import HyperModel, Hyperband
from keras_tuner import HyperParameters
from keras_tuner import Objective

class MyHyperModel(HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(keras.layers.Dense(3, activation='softmax'))  # Assuming 3 classes
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def macro_auc_objective(val_data):
    return Objective('val_macro_auc', direction='max')


import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

hypermodel = MyHyperModel()

tuner = Hyperband(
    hypermodel,
    objective=macro_auc_objective((x_val, y_val)),
    max_epochs=10,
    hyperband_iterations=2,
    directory='my_dir',
    project_name='macro_auc_tuning'
)

# Include the custom callback
macro_auc_callback = MacroAUC(validation_data=(x_val, y_val))

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[macro_auc_callback])
