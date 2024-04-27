#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:50:55 2024

@author: andrewchang
"""

import keras
import numpy as np

STM = np.load('STM_output/corpSTMnpy/Albouy2020Science_STMall.npy')

model = keras.Sequential(
    [
        keras.Input(shape=(2420,)),
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
model.summary()