#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:50:55 2024

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

from prepData import prepData_AST as prepData # if this does not work, uncomment the sections below

os.environ["KERAS_BACKEND"] = "tensorflow"

# %% build model
n_feat = 768
n_target = 6

class hyperModel_drop(kt.HyperModel):

    def build(self, hp):
        # tf.keras.backend.clear_session()
        gc.collect()
        model = keras.Sequential()
        model.add(keras.Input(shape=(n_feat,)))
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu"]),
                    kernel_regularizer=keras.regularizers.L1(l1=hp.Float(f"L1_{i}", min_value=1e-12, max_value=1e-6, sampling="log"))
                    )
                )
            model.add(
                layers.Dropout(
                    rate=hp.Float(f"drop_{i}", min_value=0, max_value=0.1, sampling="linear")
                    )
                )
    
        model.add(layers.Dense(n_target, activation="softmax"))
        
        ROC_AUC = keras.metrics.AUC(
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
        
        macroF1 = keras.metrics.F1Score(average="macro", threshold=None, name="macro_f1_score", dtype=None)
        
        learning_rate = hp.Float("lr", min_value=1e-7, max_value=1e-4, sampling="log")
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                gradient_accumulation_steps=8
                ),
            loss="categorical_focal_crossentropy",
            metrics=[ROC_AUC, macroF1],
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
        # tf.keras.backend.clear_session()
        gc.collect()
        model = keras.Sequential()
        model.add(keras.Input(shape=(n_feat,)))
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu"]),
                    kernel_regularizer=keras.regularizers.L1(l1=hp.Float(f"L1_{i}", min_value=1e-12, max_value=1e-6, sampling="log"))
                    )
                )
            model.add(layers.LayerNormalization())
    
        model.add(layers.Dense(n_target, activation="softmax"))
        
        ROC_AUC = keras.metrics.AUC(
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
        
        macroF1 = keras.metrics.F1Score(average="macro", threshold=None, name="macro_f1_score", dtype=None)
        
        learning_rate = hp.Float("lr", min_value=1e-7, max_value=1e-4, sampling="log")
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                gradient_accumulation_steps=8
                ),
            loss="categorical_focal_crossentropy",
            metrics=[ROC_AUC, macroF1],
        )
        return model
    
    def fit(self, hp, model, dataset, validation_data=None, **kwargs):
        return model.fit(
            dataset,
            shuffle=True,
            validation_data=validation_data,
            **kwargs,
        )

# %% set the tuner

if sys.argv[1]=='0':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = False)
    hm = hyperModel_drop()
    directory = "model/AST/MLP_corpora_categories/Dropout/ROC-AUC"
    objective="val_auc"
elif sys.argv[1]=='1':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = False)
    hm = hyperModel_LN()
    directory = "model/AST/MLP_corpora_categories/LayerNormalization/ROC-AUC"
    objective="val_auc"
elif sys.argv[1]=='2':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = False)
    hm = hyperModel_drop()
    directory = "model/AST/MLP_corpora_categories/Dropout/macroF1"
    objective=kt.Objective("val_macro_f1_score", direction="max")
elif sys.argv[1]=='3':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = False)
    hm = hyperModel_LN()
    directory = "model/AST/MLP_corpora_categories/LayerNormalization/macroF1"
    objective=kt.Objective("val_macro_f1_score", direction="max")
elif sys.argv[1]=='4':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = True)
    hm = hyperModel_drop()
    directory = "model/AST/MLP_corpora_categories/Dropout/ROC-AUC/downsample"
    objective="val_auc"
elif sys.argv[1]=='5':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = True)
    hm = hyperModel_LN()
    directory = "model/AST/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample"
    objective="val_auc"
elif sys.argv[1]=='6':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = True)
    hm = hyperModel_drop()
    directory = "model/AST/MLP_corpora_categories/Dropout/macroF1/downsample"
    objective=kt.Objective("val_macro_f1_score", direction="max")
elif sys.argv[1]=='7':
    train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(ds_nontonal_speech = True)
    hm = hyperModel_LN()
    directory = "model/AST/MLP_corpora_categories/LayerNormalization/macroF1/downsample"
    objective=kt.Objective("val_macro_f1_score", direction="max")
    
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# Prepare a directory to store all the checkpoints.
checkpoint_dir = directory+"/ckpt/"+time_stamp
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

## Disable GPU (not enough usage)
# Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# with strategy.scope():
    
    
tuner = kt.BayesianOptimization(
    hypermodel=hm,
    objective=objective,
    num_initial_points=10,
    max_trials=40,
    executions_per_trial=3,
    seed=23,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    overwrite=True,
    directory=directory,
    project_name="MLP_"+time_stamp,
    )
tuner.search_space_summary()

tuner.search(
    train_dataset, 
    epochs=2, 
    validation_data=val_dataset,
    # callbacks=[
    #     keras.callbacks.EarlyStopping(
    #         monitor=objective, 
    #         mode="max",
    #         patience=5,
    #         verbose=1,
    #         ),
    #     keras.callbacks.ModelCheckpoint(
    #         filepath=checkpoint_dir + "/ckpt-{epoch}.keras",
    #         save_freq="epoch",
    #         ),
    #     ]
    )


# %% retrain the best model
tf.keras.backend.clear_session()
gc.collect()

retrain_dataset = train_dataset.concatenate(val_dataset)

n_best_model = 3
best_hps = tuner.get_best_hyperparameters(n_best_model)
for n in range(n_best_model):
    best_model = hm.build(best_hps[n])
    best_model.fit(retrain_dataset)
    
    saving_path = directory+"/"+"MLP_"+time_stamp+"/best_model"+str(n)+".keras"
    best_model.save(saving_path)