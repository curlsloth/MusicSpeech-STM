#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:54:42 2024

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
import gc
import glob
import scipy.io

os.environ["KERAS_BACKEND"] = "tensorflow"

# %% load data

# def prepData_plotSTM():
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


mean_STM_list = []
for t in range(len(target.unique())):
    mean_STM_list.append(np.mean(STM_all[target==t,:], axis = 0).reshape(20,121))


# get the STM axes
df = pd.read_csv('metaTables/metaData_BibleTTS-hausa.csv',index_col=0)
survey_file = df['mat_filename'][0].replace('MATs','Survey').replace('mat_wl4','params').replace('MS2024','Params')
survey_data = scipy.io.loadmat(survey_file)
x_axis = survey_data['Params'][0]['x_axis'][0][0]
y_axis = survey_data['Params'][0]['y_axis'][0][0]

xrange = [-15, 15]
yrange = [0,7.2]

x_ds_factor = 1
y_ds_factor = 2

xmin = np.argmin(np.abs(x_axis - xrange[0]))
xmax = np.argmin(np.abs(x_axis - xrange[1]))
ymin = np.argmin(np.abs(y_axis - yrange[0]))
ymax = np.argmin(np.abs(y_axis - yrange[1]))
x_axis_small = x_axis[xmin:xmax+1:x_ds_factor]
y_axis_small = y_axis[ymin:ymax+1:y_ds_factor]

# %% plot mean STMs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors

with plt.style.context('seaborn-v0_8-poster'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    extent = [x_axis_small.min(), x_axis_small.max(), y_axis_small.min(), y_axis_small.max()]
    norm = mcolors.LogNorm(vmin=0.7*1e-1, vmax=1)
    cmap='jet'
    
    # Plot the first image in the top-left subplot
    im0 = ax[0, 0].imshow(mean_STM_list[0], aspect=2.5, origin='lower', extent=extent, norm=norm, cmap=cmap)
    ax[0, 0].set_title('Speech: nontonal')
    # ax[0, 0].set_xlabel('temporal modulation (Hz)')
    # ax[0, 0].set_ylabel('spectral modulation (cyc/oct)')
    im1 = ax[1, 0].imshow(mean_STM_list[1], aspect=2.5, origin='lower', extent=extent, norm=norm, cmap=cmap)
    ax[1, 0].set_title('Speech: tonal')
    ax[1, 0].set_xlabel('temporal modulation (Hz)')
    ax[1, 0].set_ylabel('spectral modulation (cyc/oct)')
    
    im2 = ax[0, 1].imshow(mean_STM_list[2], aspect=2.5, origin='lower', extent=extent, norm=norm, cmap=cmap)
    ax[0, 1].set_title('Music: vocal')
    # ax[0, 1].set_xlabel('temporal modulation (Hz)')
    # ax[0, 1].set_ylabel('spectral modulation (cyc/oct)')
    im3 = ax[1, 1].imshow(mean_STM_list[3], aspect=2.5, origin='lower', extent=extent, norm=norm, cmap=cmap)
    ax[1, 1].set_title('Music: nonvocal')
    # ax[1, 1].set_xlabel('temporal modulation (Hz)')
    # ax[1, 1].set_ylabel('spectral modulation (cyc/oct)')
    
    im4 = ax[0, 2].imshow(mean_STM_list[4], aspect=2.5, origin='lower', extent=extent, norm=norm, cmap=cmap)
    ax[0, 2].set_title('Env: urban')
    # ax[0, 2].set_xlabel('temporal modulation (Hz)')
    # ax[0, 2].set_ylabel('spectral modulation (cyc/oct)')
    im5 = ax[1, 2].imshow(mean_STM_list[5], aspect=2.5, origin='lower', extent=extent, norm=norm, cmap=cmap)
    ax[1, 2].set_title('Env: wildlife')
    # ax[1, 2].set_xlabel('temporal modulation (Hz)')
    # ax[1, 2].set_ylabel('spectral modulation (cyc/oct)')
    
    
    cbar_ax = fig.add_axes([0.9, 0.2, 0.01, 0.6]) 
    cbar = plt.colorbar(im1, cax=cbar_ax)
    
    # Add a single colorbar on the top side
    # cbar = fig.colorbar(im5, ax=cbar_ax, orientation='horizontal', fraction=0.04, pad=0.2)
    cbar.set_label('normalized power')
    # cbar.set_ticks(ticks=[0.1, 0.5, 1], labels=['0.1', '0.5', '1'])
    cbar.set_ticks([0.1, 0.5, 1])
    # cbar.set_ticklabels(['0.1', '0.5', '1'])
    
    # Adjust layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect parameter to fit the colorbar
    # plt.tight_layout()
    
    # plt.tight_layout()
    plt.show()
    fig.savefig('categorical_mean_STM20240906.png')
