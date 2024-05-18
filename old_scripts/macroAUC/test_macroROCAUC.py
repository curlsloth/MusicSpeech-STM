#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:14:04 2024

@author: andrewchang
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from keras_tuner import HyperModel, Hyperband, Objective
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import Callback
import keras_tuner as kt
import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.california_housing.load_data()
y_train = (y_train<100000)*1
y_test = (y_test<100000)*1


class MacroRocAuc(keras.metrics.Metric):
    def __init__(self, validation_data, **kwargs):
        super().__init__(name='macro_roc_auc', **kwargs)
        self.validation_data = validation_data
        self.macro_roc_auc = 0
        val_data, val_labels = self.validation_data
        val_predictions = self.model.predict(val_data)
        val_labels_binary = to_categorical(np.argmax(val_labels, axis=1), num_classes=val_labels.shape[1])

    def update_state(self, epoch, val_predictions, val_labels_binary, val_labels, logs=None):
        
        roc_auc_scores = []
        for i in range(val_labels.shape[1]):
            roc_auc = roc_auc_score(val_labels_binary[:, i], val_predictions[:, i])
            roc_auc_scores.append(roc_auc)

        self.macro_roc_auc = np.mean(roc_auc_scores)
        print(f'\nEpoch {epoch + 1}: Macro ROC AUC: {self.macro_roc_auc:.4f}')
        logs['macro_roc_auc'] = self.macro_roc_auc

    def result(self):
        return self.macro_roc_auc

class MyHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(hp.Int('units2', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[MacroRocAuc()])
        return model

def run_tuner():
    tuner = kt.BayesianOptimization(
        MyHyperModel(),
        objective=kt.Objective('val_macro_roc_auc', direction='max'),
        max_trials=10,
        directory='my_dir',
        project_name='macro_roc_auc'
    )

    # macro_roc_auc_callback = MacroRocAucCallback(validation_data=(X_test, y_test))

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"The best hyperparameters are: {best_hps}")

if __name__ == "__main__":
    # Example data (X_train, X_test, y_train, y_test) should be defined

    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, num_classes=np.unique(y_train).shape[0])
    y_test = to_categorical(y_test, num_classes=np.unique(y_test).shape[0])

    # Run the tuner
    run_tuner()
