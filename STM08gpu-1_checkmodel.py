#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:55:40 2024

@author: andrewchang
"""

import numpy as np
import pandas as pd
import keras
from keras import layers
import keras_tuner as kt
import datetime
import os
import tensorflow as tf
import sys
import gc

# %% build model
class hyperModel_drop(kt.HyperModel):

    def build(self, hp):
        tf.keras.backend.clear_session()
        gc.collect()
        model = keras.Sequential()
        model.add(keras.Input(shape=(n_feat,)))
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 8)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=16, max_value=512, step=16),
                    activation=hp.Choice("activation", ["relu"]),
                    kernel_regularizer=keras.regularizers.L1(l1=hp.Float(f"L1_{i}", min_value=1e-8, max_value=1e-1, sampling="log"))
                    )
                )
            model.add(
                layers.Dropout(
                    rate=hp.Float(f"drop_{i}", min_value=0, max_value=0.5, sampling="linear")
                    )
                )
    
        model.add(layers.Dense(n_target, activation="softmax"))
        
        multi_label_ROC_AUC = keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=False, # only set to True when dealing with music genres
            label_weights=None,
            from_logits=False,
        )
        learning_rate = hp.Float("lr", min_value=1e-8, max_value=1e-2, sampling="log")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_focal_crossentropy",
            metrics=[multi_label_ROC_AUC, 'f1_score'],
        )
        return model
    
    def fit(self, hp, model, dataset, validation_data=None, **kwargs):
        return model.fit(
            dataset,
            shuffle=True,
            validation_data=validation_data,
            **kwargs,
        )

class hyperModel_LN(kt.HyperModel):

    def build(self, hp):
        tf.keras.backend.clear_session()
        gc.collect()
        model = keras.Sequential()
        model.add(keras.Input(shape=(n_feat,)))
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 8)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=16, max_value=512, step=16),
                    activation=hp.Choice("activation", ["relu"]),
                    kernel_regularizer=keras.regularizers.L1(l1=hp.Float(f"L1_{i}", min_value=1e-8, max_value=1e-1, sampling="log"))
                    )
                )
            model.add(layers.LayerNormalization())
    
        model.add(layers.Dense(n_target, activation="softmax"))
        
        multi_label_ROC_AUC = keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=False, # only set to True when dealing with music genres
            label_weights=None,
            from_logits=False,
        )
        learning_rate = hp.Float("lr", min_value=1e-8, max_value=1e-2, sampling="log")
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                gradient_accumulation_steps=8
                ),
            loss="categorical_focal_crossentropy",
            metrics=[multi_label_ROC_AUC, 'f1_score'],
        )
        return model
    
    def fit(self, hp, model, dataset, validation_data=None, **kwargs):
        return model.fit(
            dataset,
            shuffle=True,
            validation_data=validation_data,
            **kwargs,
        )

# %%
objective = kt.Objective("val_auc", "max")

oracle = kt.Oracle(objective = objective)
n_feat = 2420
n_target=5


tuner = kt.Tuner(
    oracle = oracle,
    hypermodel=hyperModel_drop(),
    directory="model/MLP_corpora_categories/old/Dropout",
    project_name="MLP_2024-05-06_03-51"
    )

best_model_list = tuner.get_best_models(num_models=5)
tuner.results_summary(num_trials=5)