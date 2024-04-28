#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:50:55 2024

@author: andrewchang
"""

import keras
import numpy as np
from keras import layers
import keras_tuner as kt

STM = np.load('STM_output/corpSTMnpy/Albouy2020Science_STMall.npy')

def init_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(STM.shape[1],)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu", kernel_regularizer='l1'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(5, activation="sigmoid"),
        ]
    )
    return model
model = init_model()


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(STM.shape[1],)))
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units_1", min_value=32, max_value=512, step=32),
            activation="relu",
            )
        )
    model.add(
        layers.Dropout(
            rate=hp.Float("drop_1", min_value=0, max_value=0.5, sampling="linear")
            )
        )
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
build_model(kt.HyperParameters())

hp = kt.HyperParameters()
print(hp.Int("units", min_value=32, max_value=512, step=32))


tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    seed=23,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)
tuner.search_space_summary()