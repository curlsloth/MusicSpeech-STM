#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:49:24 2024

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
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    # % split data
    
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
    batch_size = 256

    train_dataset = train_dataset.shuffle(buffer_size=sum(train_ind), seed=23).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=sum(val_ind), seed=23).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(batch_size)
    
    n_feat = STM_all.shape[1]
    n_target = len(target.unique())
    
   
    if len(target) != len(STM_all):
        print("STM data and meta data mismatched!")
    elif sum(train_ind)+sum(val_ind)+sum(test_ind) != len(STM_all):
        print("Data split wrong")
    else:
        print("Good to go!")
        
    return train_dataset, val_dataset, test_dataset, n_feat, n_target

# print(f"***Training label ratio:\n{100*y_train.value_counts()/len(y_train)}\n")
# print(f"***Validating label ratio:\n{100*y_valid.value_counts()/len(y_valid)}\n")
# print(f"***Testing label ratio:\n{100*y_test.value_counts()/len(y_test)}\n")



# %% Prepare data
_, _, test_dataset, n_feat, n_target = prepData()

model = keras.saving.load_model("model/MLP_corpora_categories/Dropout/MLP_2024-05-14_19-00/best_model0.keras")

model.compile(metrics=['auc','f1_score','accuracy','precision','recall', 'binary_accuracy'])

evaluation = model.evaluate(test_dataset)

# %%
from sklearn.metrics import classification_report
y_pred = model.predict(test_dataset)

y_pred_argmax = list(np.argmax(y_pred, axis=1))


def extract_target(x, y):
    return y

# Apply the extraction function to the dataset
y_dataset = test_dataset.map(lambda x, y: extract_target(x, y))

# Iterate over the y dataset to see the extracted y values
y_true = []
for yi in y_dataset:
    y_true+= list(np.argmax(yi.numpy(), axis=1))


print(classification_report(y_true, y_pred_argmax))