#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:50:55 2024

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

os.environ["KERAS_BACKEND"] = "tensorflow"

# %% prepData
def prepData():
    # % load STM data
    corpus_speech_list = ['BibleTTS/akuapem-twi',
        'BibleTTS/asante-twi',
        'BibleTTS/ewe',
        'BibleTTS/hausa',
        'BibleTTS/lingala',
        'BibleTTS/yoruba',
        'Buckeye',
        'EUROM',
        'HiltonMoser2022_speech',
        'LibriSpeech',
        # 'LibriVox',
        'MediaSpeech/AR',
        'MediaSpeech/ES',
        'MediaSpeech/FR',
        'MediaSpeech/TR',
        'MozillaCommonVoice/ab',
        'MozillaCommonVoice/ar',
        'MozillaCommonVoice/ba',
        'MozillaCommonVoice/be',
        'MozillaCommonVoice/bg',
        'MozillaCommonVoice/bn',
        'MozillaCommonVoice/br',
        'MozillaCommonVoice/ca',
        'MozillaCommonVoice/ckb',
        'MozillaCommonVoice/cnh',
        'MozillaCommonVoice/cs',
        'MozillaCommonVoice/cv',
        'MozillaCommonVoice/cy',
        'MozillaCommonVoice/da',
        'MozillaCommonVoice/de',
        'MozillaCommonVoice/dv',
        'MozillaCommonVoice/el',
        'MozillaCommonVoice/en',
        'MozillaCommonVoice/eo',
        'MozillaCommonVoice/es',
        'MozillaCommonVoice/et',
        'MozillaCommonVoice/eu',
        'MozillaCommonVoice/fa',
        'MozillaCommonVoice/fi',
        'MozillaCommonVoice/fr',
        'MozillaCommonVoice/fy-NL',
        'MozillaCommonVoice/ga-IE',
        'MozillaCommonVoice/gl',
        'MozillaCommonVoice/gn',
        'MozillaCommonVoice/hi',
        'MozillaCommonVoice/hu',
        'MozillaCommonVoice/hy-AM',
        'MozillaCommonVoice/id',
        'MozillaCommonVoice/ig',
        'MozillaCommonVoice/it',
        'MozillaCommonVoice/ja',
        'MozillaCommonVoice/ka',
        'MozillaCommonVoice/kab',
        'MozillaCommonVoice/kk',
        'MozillaCommonVoice/kmr',
        'MozillaCommonVoice/ky',
        'MozillaCommonVoice/lg',
        'MozillaCommonVoice/lt',
        'MozillaCommonVoice/ltg',
        'MozillaCommonVoice/lv',
        'MozillaCommonVoice/mhr',
        'MozillaCommonVoice/ml',
        'MozillaCommonVoice/mn',
        'MozillaCommonVoice/mt',
        'MozillaCommonVoice/nan-tw',
        'MozillaCommonVoice/nl',
        'MozillaCommonVoice/oc',
        'MozillaCommonVoice/or',
        'MozillaCommonVoice/pl',
        'MozillaCommonVoice/pt',
        'MozillaCommonVoice/ro',
        'MozillaCommonVoice/ru',
        'MozillaCommonVoice/rw',
        'MozillaCommonVoice/sr',
        'MozillaCommonVoice/sv-SE',
        'MozillaCommonVoice/sw',
        'MozillaCommonVoice/ta',
        'MozillaCommonVoice/th',
        'MozillaCommonVoice/tr',
        'MozillaCommonVoice/tt',
        'MozillaCommonVoice/ug',
        'MozillaCommonVoice/uk',
        'MozillaCommonVoice/ur',
        'MozillaCommonVoice/uz',
        'MozillaCommonVoice/vi',
        'MozillaCommonVoice/yo',
        'MozillaCommonVoice/yue',
        'MozillaCommonVoice/zh-CN',
        'MozillaCommonVoice/zh-TW',
        'primewords_chinese',
        'room_reader',
        'SpeechClarity',
        'TAT-Vol2',
        'thchs30',
        'TIMIT',
        'TTS_Javanese',
        'zeroth_korean'
    ]
    
    corpus_music_list = [
        'IRMAS',
        'Albouy2020Science',
        'CD',
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC'] # no SONYC_augmented
    
    # sort the corpora lists to make sure the order is replicable
    corpus_speech_list.sort()
    corpus_music_list.sort()
    corpus_env_list.sort()
    
    corpus_list_all = corpus_speech_list+corpus_music_list+corpus_env_list 
    
    for corp in corpus_list_all:
        filename = 'vggish_output/embeddings/'+corp.replace('/', '-')+'_vggishEmbeddings.npy'
        if 'emb_all' not in locals():
            emb_all = np.load(filename)
        else:
            emb_all = np.vstack((emb_all, np.load(filename)))
        print(filename)
        
    # %% load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    # %% split data
    
    target = all_corp_df['corpus_type']
    
    target.replace({
        'speech: non-tonal':0,
        'speech: tonal':1,
        'music: vocal':2,
        'music: non-vocal':3,
        'env':4,
        },
        inplace=True)
    
    y = keras.utils.to_categorical(target, num_classes=5)
    
    data_split = all_corp_df['10fold_labels']

    
    train_ind = data_split<8
    val_ind = data_split==8
    test_ind = data_split==9
    
    train_dataset = tf.data.Dataset.from_tensor_slices((emb_all[train_ind,:], y[train_ind,:]))
    val_dataset = tf.data.Dataset.from_tensor_slices((emb_all[val_ind,:], y[val_ind,:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((emb_all[test_ind,:], y[test_ind,:]))
    
    # shuffle and then batch
    batch_size = 256

    train_dataset = train_dataset.shuffle(buffer_size=sum(train_ind), seed=23).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=sum(val_ind), seed=23).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(batch_size)
    
    n_feat = emb_all.shape[1]
    n_target = len(target.unique())
    
   
    if len(target) != len(emb_all):
        print("Embedding data and meta data mismatched!")
    elif sum(train_ind)+sum(val_ind)+sum(test_ind) != len(emb_all):
        print("Data split wrong")
    else:
        print("Good to go!")
        
    return train_dataset, val_dataset, test_dataset, n_feat, n_target

# print(f"***Training label ratio:\n{100*y_train.value_counts()/len(y_train)}\n")
# print(f"***Validating label ratio:\n{100*y_valid.value_counts()/len(y_valid)}\n")
# print(f"***Testing label ratio:\n{100*y_test.value_counts()/len(y_test)}\n")



# %% Prepare data
train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData()

# %% build model
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
                    units=hp.Int(f"units_{i}", min_value=200, max_value=600, step=50),
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
                    units=hp.Int(f"units_{i}", min_value=200, max_value=600, step=50),
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
    hm = hyperModel_drop()
    directory = "model/VGGish/MLP_corpora_categories/Dropout/ROC-AUC"
    objective="val_auc"
elif sys.argv[1]=='1':
    hm = hyperModel_LN()
    directory = "model/VGGish/MLP_corpora_categories/LayerNormalization/ROC-AUC"
    objective="val_auc"
elif sys.argv[1]=='2':
    hm = hyperModel_drop()
    directory = "model/VGGish/MLP_corpora_categories/Dropout/macroF1"
    objective=kt.Objective("val_macro_f1_score", direction="max")
elif sys.argv[1]=='3':
    hm = hyperModel_LN()
    directory = "model/VGGish/MLP_corpora_categories/LayerNormalization/macroF1"
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