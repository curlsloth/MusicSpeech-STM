#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:14:04 2024

@author: andrewchang
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras_tuner import HyperModel, BayesianOptimization, Objective
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras
import keras_tuner as kt

# Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.california_housing.load_data()
y_train = (y_train < 100000).astype(int)
y_test = (y_test < 100000).astype(int)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=np.unique(y_train).shape[0])
y_test = to_categorical(y_test, num_classes=np.unique(y_test).shape[0])


class MacroRocAuc(keras.metrics.Metric):
    def __init__(self, name='macro_roc_auc', **kwargs):
        super(MacroRocAuc, self).__init__(name=name, **kwargs)
        self.macro_roc_auc = self.add_weight(name='macro_roc_auc', initializer='zeros', dtype=tf.float32)
        self.roc_auc_scores = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true and y_pred are converted to numpy arrays
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        # y_true = y_true.numpy()
        # y_pred = y_pred.numpy()

        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle case where there's only one class present in y_true
            roc_auc = 0.5

        self.roc_auc_scores.append(roc_auc)
        self.macro_roc_auc.assign(tf.reduce_mean(self.roc_auc_scores))

    def result(self):
        return self.macro_roc_auc

    def reset_states(self):
        self.macro_roc_auc.assign(0.0)
        self.roc_auc_scores = []


class MyHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(hp.Int('units2', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[MacroRocAuc()])
        return model


def run_tuner():
    tuner = BayesianOptimization(
        MyHyperModel(),
        objective=Objective('val_macro_roc_auc', direction='max'),
        max_trials=10,
        directory='my_dir',
        project_name='macro_roc_auc'
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"The best hyperparameters are: {best_hps}")


if __name__ == "__main__":
    run_tuner()
