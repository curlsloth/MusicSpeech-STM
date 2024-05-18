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
def prepData(feat_type):
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
    
    corpus_env_list = ['SONYC']
    
    # sort the corpora lists to make sure the order is replicable
    corpus_speech_list.sort()
    corpus_music_list.sort()
    corpus_env_list.sort()
    
    corpus_list_all = corpus_speech_list+corpus_music_list+corpus_env_list 
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    # % split data
    
    target = all_corp_df['corpus_type']
    data_split = all_corp_df['10fold_labels']
    
    target.replace({
        'speech: non-tonal':0,
        'speech: tonal':1,
        'music: vocal':2,
        'music: non-vocal':3,
        'env':4,
        },
        inplace=True)
    
    y = keras.utils.to_categorical(target, num_classes=5)
    
    # train_ind = data_split<8
    # val_ind = data_split==8
    test_ind = data_split==9
    
    batch_size = 256
    
    test_dataset, n_feat, n_target = get_data(feat_type, corpus_list_all, test_ind, y, batch_size, target)
    
    return test_dataset, n_feat, n_target

def get_data(feat_type, corpus_list_all, test_ind, y, batch_size, target):
    for corp in corpus_list_all:
        if feat_type=='STM':
            filename = 'STM_output/corpSTMnpy/'+corp.replace('/', '-')+'_STMall.npy'
        elif feat_type=='VGG':
            filename = 'vggish_output/embeddings/'+corp.replace('/', '-')+'_vggishEmbeddings.npy'
        elif feat_type=='YAM':
            filename = 'yamnet_output/embeddings/'+corp.replace('/', '-')+'_yamnetEmbeddings.npy'
        
        if 'data_all' not in locals():
            data_all = np.load(filename)
        else:
            data_all = np.vstack((data_all, np.load(filename)))
        print(filename)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((data_all[test_ind,:], y[test_ind,:]))
    test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(batch_size)
    n_feat = data_all.shape[1]
    n_target = len(target.unique())
    
    return test_dataset, n_feat, n_target


# %% Load data
test_dataset_STM, n_feat_STM, n_target = prepData(feat_type='STM')
test_dataset_VGG, n_feat_VGG, n_target = prepData(feat_type='VGG')
test_dataset_YAM, n_feat_YAM, n_target = prepData(feat_type='YAM')

# %% Load models

model_STM_dropout_F1 = keras.saving.load_model("model/STM/MLP_corpora_categories/Dropout/macroF1/MLP_2024-05-17_14-42/best_model0.keras")
model_STM_dropout_AUC = keras.saving.load_model("model/STM/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-05-17_14-42/best_model0.keras")
model_STM_LN_F1 = keras.saving.load_model("model/STM/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-05-17_14-42/best_model0.keras")
model_STM_LN_AUC = keras.saving.load_model("model/STM/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-05-17_14-42/best_model0.keras")

model_VGG_dropout_F1 = keras.saving.load_model("model/VGGish/MLP_corpora_categories/Dropout/macroF1/MLP_2024-05-17_05-16/best_model0.keras")
model_VGG_dropout_AUC = keras.saving.load_model("model/VGGish/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-05-16_08-35/best_model0.keras")
model_VGG_LN_F1 = keras.saving.load_model("model/VGGish/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-05-17_05-16/best_model0.keras")
model_VGG_LN_AUC = keras.saving.load_model("model/VGGish/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-05-16_08-35/best_model0.keras")

model_YAM_dropout_F1 = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/Dropout/macroF1/MLP_2024-05-17_05-17/best_model0.keras")
model_YAM_dropout_AUC = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-05-16_08-23/best_model0.keras")
model_YAM_LN_F1 = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-05-17_05-17/best_model0.keras")
model_YAM_LN_AUC = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-05-16_08-23/best_model0.keras")


def eval_model(model, test_dataset):
    macroF1 = keras.metrics.F1Score(average="macro", threshold=None, name="macro_f1_score", dtype=None)
    model.compile(metrics=['auc','f1_score', macroF1,'accuracy'])
    evaluation = model.evaluate(test_dataset)
    # Extract tensor values and flatten the list
    tensor_values = evaluation[2].numpy()
    flat_data = evaluation[:2] + tensor_values.tolist() + evaluation[3:]
    # Define column names
    columns = ['loss', 'ROC-AUC', 'f1: speech (nontonal)', 'f1: speech (tonal)', 'f1: music (vocal)', 'f1: music (nonvocal)', 'f1: environmental sound', 'macro_f1', 'accuracy']
    # Create DataFrame
    df = pd.DataFrame([flat_data], columns=columns)
    return df

eval_STM_dropout_F1 = eval_model(model_STM_dropout_F1, test_dataset_STM)
eval_STM_dropout_F1['model'] = 'STM_dropout_F1'
eval_STM_dropout_AUC = eval_model(model_STM_dropout_AUC, test_dataset_STM)
eval_STM_dropout_AUC['model'] = 'STM_dropout_AUC'

eval_YAM_dropout_F1 = eval_model(model_YAM_dropout_F1, test_dataset_YAM)
eval_YAM_dropout_F1['model'] = 'YAM_dropout_F1'
eval_YAM_dropout_AUC = eval_model(model_YAM_dropout_AUC, test_dataset_YAM)
eval_YAM_dropout_AUC['model'] = 'YAM_dropout_AUC'

eval_VGG_dropout_F1 = eval_model(model_VGG_dropout_F1, test_dataset_VGG)
eval_VGG_dropout_F1['model'] = 'VGG_dropout_F1'
eval_VGG_dropout_AUC = eval_model(model_VGG_dropout_AUC, test_dataset_VGG)
eval_VGG_dropout_AUC['model'] = 'VGG_dropout_AUC'

eval_STM_LN_F1 = eval_model(model_STM_LN_F1, test_dataset_STM)
eval_STM_LN_F1['model'] = 'STM_LN_F1'
eval_STM_LN_AUC = eval_model(model_STM_LN_AUC, test_dataset_STM)
eval_STM_LN_AUC['model'] = 'STM_LN_AUC'

eval_YAM_LN_F1 = eval_model(model_YAM_LN_F1, test_dataset_YAM)
eval_YAM_LN_F1['model'] = 'YAM_LN_F1'
eval_YAM_LN_AUC = eval_model(model_YAM_LN_AUC, test_dataset_YAM)
eval_YAM_LN_AUC['model'] = 'YAM_LN_AUC'

eval_VGG_LN_F1 = eval_model(model_VGG_LN_F1, test_dataset_VGG)
eval_VGG_LN_F1['model'] = 'VGG_LN_F1'
eval_VGG_LN_AUC = eval_model(model_VGG_LN_AUC, test_dataset_VGG)
eval_VGG_LN_AUC['model'] = 'VGG_LN_AUC'

df_eval = pd.concat([
    eval_STM_dropout_F1,eval_STM_dropout_AUC,
    eval_YAM_dropout_F1,eval_YAM_dropout_AUC,
    eval_VGG_dropout_F1,eval_VGG_dropout_AUC,
    eval_STM_LN_F1,eval_STM_LN_AUC,
    eval_YAM_LN_F1,eval_YAM_LN_AUC,
    eval_VGG_LN_F1,eval_VGG_LN_AUC], ignore_index=True)

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