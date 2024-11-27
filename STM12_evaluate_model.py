#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:49:24 2024

"""


import numpy as np
import pandas as pd
import keras
from keras import layers
import keras_tuner as kt
import datetime
import os
import tensorflow as tf
import gc
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


os.environ["KERAS_BACKEND"] = "tensorflow"

from prepData import prepData_STM, prepData_VGG, prepData_YAM, prepData_melspectrogram, mask_STMmatrix



def eval_model(model, test_dataset):
    # search the max F1 score across thresholds
    macroF1_list = []
    for threshold in range(5,100,5):
        macroF1_list.append(keras.metrics.F1Score(average="macro", threshold=threshold/100, name="macro_f1_score_"+str(threshold), dtype=None))
    
    ROCAUC = keras.metrics.AUC(curve="ROC", name="ROC-AUC")
    PRAUC = keras.metrics.AUC(curve="PR", name="PR-AUC")
    
    model.compile(metrics=[ROCAUC, PRAUC,'accuracy']+macroF1_list)
    evaluation = model.evaluate(test_dataset)
    max_threshold = (np.argmax(evaluation[4:])+1)*0.05
    
    df_f1 = eval_model_classF1(model, test_dataset, max_threshold)
    
    flat_data = evaluation[:3] + [max(evaluation[4:])] + [evaluation[3]] +[max_threshold]
    # Define column names
    columns = ['loss', 'ROC-AUC', 'PR-AUC', 'max_macro_f1', 'accuracy', 'max_f1_threshold']
    # Create DataFrame
    df = pd.DataFrame([flat_data], columns=columns)
    df_all = pd.concat([df, df_f1], axis=1)
    return df_all

def eval_model_classF1(model, test_dataset, threshold):
    macroF1_list = []
    macroF1_list.append(keras.metrics.F1Score(average=None, threshold=threshold))
    
    model.compile(metrics=macroF1_list)
    evaluation = model.evaluate(test_dataset)
    columns = ['speech: nontonal', 'speech: tonal', 'music: vocal', 'music: nonvocal', 'env: urban', 'env: wildlife']
    df = pd.DataFrame([list(evaluation[1].numpy())], columns=columns)
    return df

# %% Load data
_, _, test_dataset_STM, n_feat_STM, n_target = prepData_STM()
_, _, test_dataset_VGG, n_feat_VGG, n_target = prepData_VGG()
_, _, test_dataset_YAM, n_feat_YAM, n_target = prepData_YAM()
_, _, test_dataset_mel, n_feat_mel, n_target = prepData_melspectrogram()

_, _, test_dataset_STM_ds, n_feat_STM, n_target = prepData_STM(ds_nontonal_speech=True)
_, _, test_dataset_VGG_ds, n_feat_VGG, n_target = prepData_VGG(ds_nontonal_speech=True)
_, _, test_dataset_YAM_ds, n_feat_YAM, n_target = prepData_YAM(ds_nontonal_speech=True)
_, _, test_dataset_mel_ds, n_feat_mel, n_target = prepData_melspectrogram(ds_nontonal_speech=True)

# %% Load models

model_STM_dropout_F1 = keras.saving.load_model("model/STM/MLP_corpora_categories/Dropout/macroF1/MLP_2024-08-31_22-11/best_model0.keras")
model_STM_dropout_AUC = keras.saving.load_model("model/STM/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-08-31_22-10/best_model0.keras")
model_STM_LN_F1 = keras.saving.load_model("model/STM/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-08-31_22-10/best_model0.keras")
model_STM_LN_AUC = keras.saving.load_model("model/STM/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-08-31_22-11/best_model0.keras")

model_VGG_dropout_F1 = keras.saving.load_model("model/VGGish/MLP_corpora_categories/Dropout/macroF1/MLP_2024-08-20_22-39/best_model0.keras")
model_VGG_dropout_AUC = keras.saving.load_model("model/VGGish/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-08-20_22-39/best_model0.keras")
model_VGG_LN_F1 = keras.saving.load_model("model/VGGish/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-08-20_22-39/best_model0.keras")
model_VGG_LN_AUC = keras.saving.load_model("model/VGGish/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-08-20_22-39/best_model0.keras")

model_YAM_dropout_F1 = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/Dropout/macroF1/MLP_2024-08-20_22-40/best_model0.keras")
model_YAM_dropout_AUC = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-08-20_22-40/best_model0.keras")
model_YAM_LN_F1 = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-08-20_22-40/best_model0.keras")
model_YAM_LN_AUC = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-08-20_22-40/best_model0.keras")

# model_mel_dropout_F1 = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/Dropout/macroF1/MLP_2024-11-24_20-48/best_model0.keras")
# model_mel_dropout_AUC = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-11-24_20-25/best_model0.keras")
# model_mel_LN_F1 = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-11-25_02-17/best_model0.keras")
# model_mel_LN_AUC = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-11-24_20-25/best_model0.keras")

model_mel_dropout_F1 = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/Dropout/macroF1/MLP_2024-11-26_17-56/best_model0.keras")
model_mel_dropout_AUC = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/Dropout/ROC-AUC/MLP_2024-11-26_17-56/best_model0.keras")
model_mel_LN_F1 = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/LayerNormalization/macroF1/MLP_2024-11-26_17-56/best_model0.keras")
model_mel_LN_AUC = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/LayerNormalization/ROC-AUC/MLP_2024-11-26_17-56/best_model0.keras")




model_STM_dropout_F1_ds = keras.saving.load_model("model/STM/MLP_corpora_categories/Dropout/macroF1/downsample/MLP_2024-08-31_22-51/best_model0.keras")
model_STM_dropout_AUC_ds = keras.saving.load_model("model/STM/MLP_corpora_categories/Dropout/ROC-AUC/downsample/MLP_2024-08-31_22-10/best_model0.keras")
model_STM_LN_F1_ds = keras.saving.load_model("model/STM/MLP_corpora_categories/LayerNormalization/macroF1/downsample/MLP_2024-08-31_23-02/best_model0.keras")
model_STM_LN_AUC_ds = keras.saving.load_model("model/STM/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample/MLP_2024-08-31_22-10/best_model0.keras")

model_VGG_dropout_F1_ds = keras.saving.load_model("model/VGGish/MLP_corpora_categories/Dropout/macroF1/downsample/MLP_2024-08-20_22-39/best_model0.keras")
model_VGG_dropout_AUC_ds = keras.saving.load_model("model/VGGish/MLP_corpora_categories/Dropout/ROC-AUC/downsample/MLP_2024-08-20_22-39/best_model0.keras")
model_VGG_LN_F1_ds = keras.saving.load_model("model/VGGish/MLP_corpora_categories/LayerNormalization/macroF1/downsample/MLP_2024-08-20_22-39/best_model0.keras")
model_VGG_LN_AUC_ds = keras.saving.load_model("model/VGGish/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample/MLP_2024-08-20_22-39/best_model0.keras")

model_YAM_dropout_F1_ds = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/Dropout/macroF1/downsample/MLP_2024-08-20_22-40/best_model0.keras")
model_YAM_dropout_AUC_ds = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/Dropout/ROC-AUC/downsample/MLP_2024-08-20_22-40/best_model0.keras")
model_YAM_LN_F1_ds = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/LayerNormalization/macroF1/downsample/MLP_2024-08-20_22-40/best_model0.keras")
model_YAM_LN_AUC_ds = keras.saving.load_model("model/YAMNet/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample/MLP_2024-08-20_22-40/best_model0.keras")

# model_mel_dropout_F1_ds = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/Dropout/macroF1/downsample/MLP_2024-11-25_03-54/best_model0.keras")
# model_mel_dropout_AUC_ds = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/Dropout/ROC-AUC/downsample/MLP_2024-11-25_02-45/best_model0.keras")
# model_mel_LN_F1_ds = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/LayerNormalization/macroF1/downsample/MLP_2024-11-25_04-38/best_model0.keras")
# model_mel_LN_AUC_ds = keras.saving.load_model("model/melspectrogram/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample/MLP_2024-11-25_03-11/best_model0.keras")

# model_mel_dropout_F1_ds = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/Dropout/macroF1/downsample/MLP_2024-11-26_18-47/best_model0.keras")
# model_mel_dropout_AUC_ds = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/Dropout/ROC-AUC/downsample/MLP_2024-11-26_17-59/best_model0.keras")
# model_mel_LN_F1_ds = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/LayerNormalization/macroF1/downsample/MLP_2024-11-26_18-51/best_model0.keras")
# model_mel_LN_AUC_ds = keras.saving.load_model("model/melspectrogram_norm/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample/MLP_2024-11-26_18-38/best_model0.keras")

model_mel_dropout_F1_ds = keras.saving.load_model("model/melspectrogram_norm_clipnorm/MLP_corpora_categories/Dropout/macroF1/downsample/MLP_2024-11-26_21-32/best_model0.keras")
model_mel_dropout_AUC_ds = keras.saving.load_model("model/melspectrogram_norm_clipnorm/MLP_corpora_categories/Dropout/ROC-AUC/downsample/MLP_2024-11-26_21-33/best_model0.keras")
model_mel_LN_F1_ds = keras.saving.load_model("model/melspectrogram_norm_clipnorm/MLP_corpora_categories/LayerNormalization/macroF1/downsample/MLP_2024-11-26_21-32/best_model0.keras")
model_mel_LN_AUC_ds = keras.saving.load_model("model/melspectrogram_norm_clipnorm/MLP_corpora_categories/LayerNormalization/ROC-AUC/downsample/MLP_2024-11-26_21-32/best_model0.keras")




# %% run evaluations

# full sample size
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

eval_mel_dropout_F1 = eval_model(model_mel_dropout_F1, test_dataset_mel)
eval_mel_dropout_F1['model'] = 'mel_dropout_F1'
eval_mel_dropout_AUC = eval_model(model_mel_dropout_AUC, test_dataset_mel)
eval_mel_dropout_AUC['model'] = 'mel_dropout_AUC'


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

eval_mel_LN_F1 = eval_model(model_mel_LN_F1, test_dataset_mel)
eval_mel_LN_F1['model'] = 'mel_LN_F1'
eval_mel_LN_AUC = eval_model(model_mel_LN_AUC, test_dataset_mel)
eval_mel_LN_AUC['model'] = 'mel_LN_AUC'

# df_eval = pd.concat([
#     eval_STM_dropout_F1,eval_STM_dropout_AUC,
#     eval_STM_LN_F1,eval_STM_LN_AUC,
#     eval_YAM_dropout_F1,eval_YAM_dropout_AUC,
#     eval_YAM_LN_F1,eval_YAM_LN_AUC,
#     eval_VGG_dropout_F1,eval_VGG_dropout_AUC,
#     eval_VGG_LN_F1,eval_VGG_LN_AUC,
#     ], ignore_index=True)


# downsampled nontonal speech
eval_STM_dropout_F1_ds = eval_model(model_STM_dropout_F1_ds, test_dataset_STM_ds)
eval_STM_dropout_F1_ds['model'] = 'STM_dropout_F1_ds'
eval_STM_dropout_AUC_ds = eval_model(model_STM_dropout_AUC_ds, test_dataset_STM_ds)
eval_STM_dropout_AUC_ds['model'] = 'STM_dropout_AUC_ds'

eval_YAM_dropout_F1_ds = eval_model(model_YAM_dropout_F1_ds, test_dataset_YAM_ds)
eval_YAM_dropout_F1_ds['model'] = 'YAM_dropout_F1_ds'
eval_YAM_dropout_AUC_ds = eval_model(model_YAM_dropout_AUC_ds, test_dataset_YAM_ds)
eval_YAM_dropout_AUC_ds['model'] = 'YAM_dropout_AUC_ds'

eval_VGG_dropout_F1_ds = eval_model(model_VGG_dropout_F1_ds, test_dataset_VGG_ds)
eval_VGG_dropout_F1_ds['model'] = 'VGG_dropout_F1_ds'
eval_VGG_dropout_AUC_ds = eval_model(model_VGG_dropout_AUC_ds, test_dataset_VGG_ds)
eval_VGG_dropout_AUC_ds['model'] = 'VGG_dropout_AUC_ds'

eval_mel_dropout_F1_ds = eval_model(model_mel_dropout_F1_ds, test_dataset_mel_ds)
eval_mel_dropout_F1_ds['model'] = 'mel_dropout_F1_ds'
eval_mel_dropout_AUC_ds = eval_model(model_mel_dropout_AUC_ds, test_dataset_mel_ds)
eval_mel_dropout_AUC_ds['model'] = 'mel_dropout_AUC_ds'


eval_STM_LN_F1_ds = eval_model(model_STM_LN_F1_ds, test_dataset_STM_ds)
eval_STM_LN_F1_ds['model'] = 'STM_LN_F1_ds'
eval_STM_LN_AUC_ds = eval_model(model_STM_LN_AUC_ds, test_dataset_STM_ds)
eval_STM_LN_AUC_ds['model'] = 'STM_LN_AUC_ds'

eval_YAM_LN_F1_ds = eval_model(model_YAM_LN_F1_ds, test_dataset_YAM_ds)
eval_YAM_LN_F1_ds['model'] = 'YAM_LN_F1_ds'
eval_YAM_LN_AUC_ds = eval_model(model_YAM_LN_AUC_ds, test_dataset_YAM_ds)
eval_YAM_LN_AUC_ds['model'] = 'YAM_LN_AUC_ds'

eval_VGG_LN_F1_ds = eval_model(model_VGG_LN_F1_ds, test_dataset_VGG_ds)
eval_VGG_LN_F1_ds['model'] = 'VGG_LN_F1_ds'
eval_VGG_LN_AUC_ds = eval_model(model_VGG_LN_AUC_ds, test_dataset_VGG_ds)
eval_VGG_LN_AUC_ds['model'] = 'VGG_LN_AUC_ds'

eval_mel_LN_F1_ds = eval_model(model_mel_LN_F1_ds, test_dataset_mel_ds)
eval_mel_LN_F1_ds['model'] = 'mel_LN_F1_ds'
eval_mel_LN_AUC_ds = eval_model(model_mel_LN_AUC_ds, test_dataset_mel_ds)
eval_mel_LN_AUC_ds['model'] = 'mel_LN_AUC_ds'


df_eval = pd.concat([
    eval_STM_dropout_F1,eval_STM_dropout_AUC,
    eval_STM_LN_F1,eval_STM_LN_AUC,
    eval_YAM_dropout_F1,eval_YAM_dropout_AUC,
    eval_YAM_LN_F1,eval_YAM_LN_AUC,
    eval_VGG_dropout_F1,eval_VGG_dropout_AUC,
    eval_VGG_LN_F1,eval_VGG_LN_AUC,
    eval_STM_dropout_F1_ds,eval_STM_dropout_AUC_ds,
    eval_STM_LN_F1_ds,eval_STM_LN_AUC_ds,
    eval_YAM_dropout_F1_ds,eval_YAM_dropout_AUC_ds,
    eval_YAM_LN_F1_ds,eval_YAM_LN_AUC_ds,
    eval_VGG_dropout_F1_ds,eval_VGG_dropout_AUC_ds,
    eval_VGG_LN_F1_ds,eval_VGG_LN_AUC_ds,
    ], ignore_index=True)

# df_eval.to_csv("model/MLP_summary_all_20240912.csv", index=False)

df_mel_eval = pd.concat([
    eval_mel_dropout_F1,eval_mel_dropout_AUC,
    eval_mel_LN_F1,eval_mel_LN_AUC,
    eval_mel_dropout_F1_ds,eval_mel_dropout_AUC_ds,
    eval_mel_LN_F1_ds,eval_mel_LN_AUC_ds,
    ], ignore_index=True)

# df_mel_eval.to_csv("model/MLP_summary_melspectrogram_20241125.csv", index=False)



# %% F1 score for each class

STM_F1_class = eval_model_classF1(model_STM_LN_F1, test_dataset_STM, 0.35)


# %% Load functions for evaluating ablation models


def prepData_STM_numpy():
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
        # 'CD', # exclude CDs due to open source concern
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC', 'MacaulayLibrary']
    
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
        print(filename)
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)

    target = all_corp_df['corpus_type']
    data_split = all_corp_df['10fold_labels']
    
    target.replace({
        'speech: non-tonal':0,
        'speech: tonal':1,
        'music: vocal':2,
        'music: non-vocal':3,
        'env: urban':4,
        'env: wildlife':5,
        },
        inplace=True)
    
    return STM_all, target, data_split

def filter_STM(STM_all, target, data_split, ablation_params):
    
    mask_matrix = mask_STMmatrix(ablation_params).flatten()
    # np.random.seed(23)
    # STM_all[:, mask_matrix==1] = np.random.rand(STM_all.shape[0], np.sum(mask_matrix))
    # STM_all[:, mask_matrix==1] = 0.0 # put the filtered out regions as 0
    STM_all = STM_all[:,mask_matrix==0] # exclude the filtered out regions (Sept 6)
    del mask_matrix
    
    y = keras.utils.to_categorical(target, num_classes=len(target.unique()))
       
        
    # % split data
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

def load_ablation_keras(highlow, xcutoff, ycutoff, STM_all, target, data_split):
    # load model
    cond = 'x'+highlow+'cutoff'+str(xcutoff)+'_y'+highlow+'cutoff'+str(ycutoff)
    filename = glob.glob('model/STM/MLP_corpora_categories/LayerNormalization/macroF1/ablation/'+cond+'/**/best_model0.keras')[0]
    model = keras.saving.load_model(filename)
    
    # load data
    if highlow=='high':
        ablation_params = {
            'x_lowcutoff': None,
            'x_highcutoff': xcutoff,
            'y_lowcutoff': None,
            'y_highcutoff': ycutoff,
            }
    elif highlow=='low':
        ablation_params = {
            'x_lowcutoff': xcutoff,
            'x_highcutoff': None,
            'y_lowcutoff': ycutoff,
            'y_highcutoff': None,
            }
    # _, _, test_dataset_STM, n_feat_STM, n_target = prepData_STM(ablation_params=ablation_params)
    _, _, test_dataset_STM, n_feat_STM, n_target = filter_STM(STM_all, target, data_split, ablation_params)
    
    # run model
    df = eval_model(model, test_dataset_STM)
    df['highlow'] = highlow
    df['xcutoff'] = xcutoff
    df['ycutoff'] = ycutoff
    return df

# %% ablation models evaluation
ablation_df_list = []
STM_all, target, data_split = prepData_STM_numpy() # load the overall data one time

for highlow in ['high','low']:
    for xcutoff in range(7):
        for ycutoff in range(7):
            ablation_df_list.append(load_ablation_keras(highlow, xcutoff, ycutoff, STM_all, target, data_split))

# this name is wrong!!
pd.concat(ablation_df_list).to_csv("model/MLP_ablation_20240907.csv", index=False)

# %% plot ablation results

df = pd.read_csv("model/MLP_ablation_20240907.csv")
table_low_ROCAUC = pd.pivot_table(df[df['highlow']=='low'], values='ROC-AUC', index=['xcutoff'],  columns=['ycutoff'])
table_high_ROCAUC = pd.pivot_table(df[df['highlow']=='high'], values='ROC-AUC', index=['xcutoff'],  columns=['ycutoff'])
table_low_F1 = pd.pivot_table(df[df['highlow']=='low'], values='max_macro_f1', index=['xcutoff'],  columns=['ycutoff'])
table_high_F1 = pd.pivot_table(df[df['highlow']=='high'], values='max_macro_f1', index=['xcutoff'],  columns=['ycutoff'])


def show_values(data, ax):
    for (j, i), val in np.ndenumerate(data):
        ax.text(i,j,f'{val:.3f}'[1:],ha='center',va='center')

with plt.style.context('seaborn-v0_8-poster'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    
    centers = [0,6,0,6]
    dx, = np.diff(centers[:2])/(7-1)
    dy, = -np.diff(centers[2:])/(7-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
    
    # extent = [0, 6, 0, 6]
    cmap='plasma'
    
    # Plot ROC-AUC
    min_ROCAUC = np.min(pd.concat([table_high_ROCAUC, table_low_ROCAUC]))
    # min_ROCAUC = 0.92
    max_ROCAUC = np.max(pd.concat([table_high_ROCAUC, table_low_ROCAUC]))
    # norm_ROCAUC = mcolors.LogNorm(vmin=min_ROCAUC, vmax=max_ROCAUC, clip=True)
    
    im0 = ax[0,0].imshow(table_high_ROCAUC, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, vmin=min_ROCAUC, vmax=max_ROCAUC)
    # im0 = ax[0,0].imshow(table_high_ROCAUC, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, norm=norm_ROCAUC)
    ax[0,0].set_title('ROC-AUC: at or below the cutoffs')
    ax[0,0].set_xlabel('temporal modulation cutoff (absolute Hz)')
    ax[0,0].set_ylabel('spectral modulation cutoff (cyc/oct)')
    show_values(table_high_ROCAUC, ax[0,0])
    
    im1 = ax[0,1].imshow(table_low_ROCAUC, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, vmin=min_ROCAUC, vmax=max_ROCAUC)
    # im1 = ax[0,1].imshow(table_low_ROCAUC, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, norm=norm_ROCAUC)
    ax[0,1].set_title('ROC-AUC: greater than the cutoffs')
    ax[0,1].set_xlabel('temporal modulation cutoff (absolute Hz)')
    # ax[0,1].set_yticks([])
    ax[0,1].set_ylabel('spectral modulation cutoff (cyc/oct)')
    show_values(table_low_ROCAUC, ax[0,1])
    
    cbar_rocauc_ax = fig.add_axes([0.9, 0.6, 0.01, 0.3]) 
    cbar_rocauc = plt.colorbar(im1, cax=cbar_rocauc_ax)
    cbar_rocauc.set_label('ROC-AUC')

    
    
    # Plot F1
    min_f1 = np.min(pd.concat([table_high_F1, table_low_F1]))
    max_f1 = np.max(pd.concat([table_high_F1, table_low_F1]))
    norm_F1 = mcolors.LogNorm(vmin=min_f1, vmax=max_f1)
    
    im2 = ax[1,0].imshow(table_high_F1, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, vmin=min_f1, vmax=max_f1)
    # im2 = ax[1,0].imshow(table_high_F1, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, norm=norm_F1)
    ax[1,0].set_title('Max F1: at or below the cutoffs')
    ax[1,0].set_xlabel('temporal modulation cutoff (absolute Hz)')
    ax[1,0].set_ylabel('spectral modulation cutoff (cyc/oct)')
    show_values(table_high_F1, ax[1,0])
    
    im3 = ax[1,1].imshow(table_low_F1, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, vmin=min_f1, vmax=max_f1)
    # im3 = ax[1,1].imshow(table_low_F1, aspect='equal', origin='lower', extent=extent, cmap=cmap, interpolation=None, norm=norm_F1)
    ax[1,1].set_title('Max F1: greater than the cutoffs')
    ax[1,1].set_xlabel('temporal modulation cutoff (absolute Hz)')
    # ax[1,1].set_yticks([])
    ax[1,1].set_ylabel('spectral modulation cutoff (cyc/oct)')
    show_values(table_low_F1, ax[1,1])
    
    cbar_f1_ax = fig.add_axes([0.9, 0.1, 0.01, 0.3]) 
    cbar_f1 = plt.colorbar(im2, cax=cbar_f1_ax)
    cbar_f1.set_label('max F1')

    
    
    # Adjust layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect parameter to fit the colorbar
    plt.show()
    fig.savefig('ablation_20240907.png')
