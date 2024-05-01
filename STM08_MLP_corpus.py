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

os.environ["KERAS_BACKEND"] = "tensorflow"

# %% load STM data
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

corpus_env_list = ['SONYC', 'SONYC_augmented']

# sort the corpora lists to make sure the order is replicable
corpus_speech_list.sort()
corpus_music_list.sort()
corpus_env_list.sort()

corpus_list_all = corpus_speech_list+corpus_music_list+corpus_env_list 

for corp in corpus_list_all:
    filename = 'STM_output/corpSTMnpy/'+corp.replace('/', '-')+'_STMall.npy'
    if 'STM_all' not in locals():
        STM_all = np.load(filename)
    else:
        STM_all = np.vstack((STM_all, np.load(filename)))
        if corp == 'SONYC_augmented':
            SONYC_aug_len = np.load(filename).shape[0]
    print(filename)
    
# %% load meta data
speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)

all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)

# %% split data

target = pd.concat([all_corp_df['corpus_type'], pd.Series(['env'] * SONYC_aug_len)], 
                   ignore_index=True)

target.replace({
    'speech: non-tonal':0,
    'speech: tonal':1,
    'music: vocal':2,
    'music: non-vocal':3,
    'env':4,
    },
    inplace=True)

y = keras.utils.to_categorical(target, num_classes=5)

data_split = pd.concat([all_corp_df['10fold_labels'], pd.Series([1] * SONYC_aug_len)],
                       ignore_index=True)

train_ind = data_split<8
val_ind = data_split==8
test_ind = data_split==9

train_dataset = tf.data.Dataset.from_tensor_slices((STM_all[train_ind,:], y[train_ind,:]))
val_dataset = tf.data.Dataset.from_tensor_slices((STM_all[val_ind,:], y[val_ind,:]))
test_dataset = tf.data.Dataset.from_tensor_slices((STM_all[test_ind,:], y[test_ind,:]))

# shuffle and then batch
train_dataset = train_dataset.shuffle(buffer_size=sum(train_ind), seed=23).batch(32)
val_dataset = val_dataset.shuffle(buffer_size=sum(val_ind), seed=23).batch(32)
test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(32)


# y_train = tf.data.Dataset.from_tensor_slices(y[train_ind,:])
# y_valid = tf.data.Dataset.from_tensor_slices(y[valid_ind,:])
# y_test = tf.data.Dataset.from_tensor_slices(y[test_ind,:])

if len(target) != len(STM_all):
    print("STM data and meta data mismatched!")
elif sum(train_ind)+sum(val_ind)+sum(test_ind) != len(STM_all):
    print("Data split wrong")
else:
    print("Good to go!")

# print(f"***Training label ratio:\n{100*y_train.value_counts()/len(y_train)}\n")
# print(f"***Validating label ratio:\n{100*y_valid.value_counts()/len(y_valid)}\n")
# print(f"***Testing label ratio:\n{100*y_test.value_counts()/len(y_test)}\n")

# %% build model

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "model/MLP_corpora_categories/ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        
        model = keras.Sequential()
        model.add(keras.Input(shape=(STM_all.shape[1],)))
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 8)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=16, max_value=512, step=16),
                    activation=hp.Choice("activation", ["relu"]),
                    kernel_regularizer=keras.regularizers.L1(l1=hp.Float(f"L1_{i}", min_value=1e-8, max_value=1e-1, sampling="log")                    )
                    )
                )
            model.add(
                layers.Dropout(
                    rate=hp.Float(f"drop_{i}", min_value=0, max_value=0.5, sampling="linear")
                    )
                )
    
        model.add(layers.Dense(len(target.unique()), activation="softmax"))
        
        multi_label_ROC_AUC = keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=True,
            num_labels=5,
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

tuner = kt.BayesianOptimization(
    hypermodel=MyHyperModel(),
    objective="val_auc",
    num_initial_points=100,
    max_trials=100,
    executions_per_trial=25,
    seed=23,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    overwrite=True,
    directory="model/MLP_corpora_categories/",
    project_name="MLP_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
)
tuner.search_space_summary()

with strategy.scope():
    tuner.search(
        train_dataset, 
        epochs=2, 
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", 
                mode="max",
                patience=5,
                verbose=1,
                ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir + "/ckpt-{epoch}.keras",
                save_freq="epoch",
                ),
            ]
        )