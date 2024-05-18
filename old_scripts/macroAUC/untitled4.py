#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:59:47 2024

@author: andrewchang
"""
from keras import ops
import keras
import numpy as np
from sklearn.metrics import roc_auc_score
from keras_tuner import HyperModel, Hyperband, Objective
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import Callback
import keras_tuner

class MacroRocAuc(keras.metrics.Metric):

  def __init__(self, name='macro_roc_auc', **kwargs):
    super().__init__(name=name, **kwargs)
    self.macro_roc_auc = self.add_weight(name='macro_roc_auc', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = ops.cast(y_true, "bool")

    # values = ops.logical_and(ops.equal(y_true, True), ops.equal(y_pred, True))
    
    # val_labels_binary = to_categorical(np.argmax(y_true.numpy(), axis=1), num_classes=y_pred.shape[1])
    # roc_auc_scores = []
    # for i in range(y_pred.shape[1]):
    #     roc_auc = roc_auc_score(val_labels_binary[:, i], y_pred[:, i])
    #     roc_auc_scores.append(roc_auc)
    values = roc_auc_score(y_true, y_pred, multi_class='ovr')
    values = ops.cast(values, self.dtype)
    
    # if sample_weight is not None:
    #   sample_weight = ops.cast(sample_weight, self.dtype)
    #   values = values * sample_weight
    self.macro_roc_auc.assign_add(ops.sum(values))

  def result(self):
    return self.macro_roc_auc

  def reset_states(self):
    self.macro_roc_auc.assign(0)
    
    
def build_regressor(hp):
    model = keras.Sequential(
        [
            keras.layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            keras.layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        # Put custom metric into the metrics.
        metrics=[MacroRocAuc()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # Specify the name and direction of the objective.
    objective=keras_tuner.Objective("val_macro_roc_auc", direction="max"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_metrics",
)

(X_train, y_train), (X_test, y_test) = keras.datasets.california_housing.load_data()
y_train = (y_train<100000)*1
y_test = (y_test<100000)*1

tuner.search(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
)

tuner.results_summary()