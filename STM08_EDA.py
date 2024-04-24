#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:19:01 2024

@author: andrewchang
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump
import datetime
import sys


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


# sort the corpora lists to make sure the order is replicable
corpus_speech_list.sort()
corpus_music_list.sort()

speech_corp_list = []
for corp in corpus_speech_list:
    metafile = 'metaTables/metaData_'+corp.replace('/', '-')+'.csv'
    speech_corp_list.append(pd.read_csv(metafile,index_col=0))
speech_corp_df = pd.concat(speech_corp_list, ignore_index=True)
    
music_corp_list = []
for corp in corpus_music_list:
    metafile = 'metaTables/metaData_'+corp.replace('/', '-')+'.csv'
    df = pd.read_csv(metafile,index_col=0)
    if 'demucs_voice' in df.columns:
        df['VoiOrNot'] = df['demucs_voice']
        df.drop(columns='demucs_voice', inplace=True)
    if 'genre' in df.columns:
        df['genre_1'] = df['genre'] 
        df.drop(columns='genre', inplace=True)
    if 'Genre' in df.columns:
        df['genre_1'] = df['Genre'] 
        df.drop(columns='Genre', inplace=True)
    music_corp_list.append(df)
music_corp_df = pd.concat(music_corp_list, ignore_index=True)

speech_corp_df['gender'].replace({
    'm':'male',
    'M':'male',
    'f':'female',
    'F':'female',
    'other':np.nan,
    'O':np.nan},
    inplace=True)


speech_corp_df['totalLengCur'].sum()
music_corp_df['totalLengCur'].sum()

sum(speech_corp_df['gender']=='male')/len(speech_corp_df)
sum(speech_corp_df['gender']=='female')/len(speech_corp_df)
sum(speech_corp_df['gender'].isna())/len(speech_corp_df)

languages = speech_corp_df['LangOrInstru'].unique()
languages.sort()

genre_columns = pd.concat([music_corp_df['genre_1'], music_corp_df['genre_2'], music_corp_df['genre_3']], axis=0)
genre_columns.replace({
    np.nan:'unlabeled',
    'Hiphop':'Hip-Hop',
    'hiphop':'Hip-Hop',
    'International':'World',
    'Soul-RnB':'Soul_RnB',
    'metal_punk': 'rock'
    },
    inplace=True)
genre_columns = genre_columns.apply(lambda x: x.title())
genres = genre_columns.unique()
genres.sort()

# %% try t-SNE

from joblib import load
pipeline = load('model/allSTM_pca-pipeline_2024-04-22_08-43.joblib')
pca = pipeline['incrementalpca']

import matplotlib.pyplot as plt

plt.plot(list(range(1,len(pca.explained_variance_ratio_[:5000])+1)), np.cumsum(pca.explained_variance_ratio_[:5000]))
plt.show()

for corp in corpus_speech_list+corpus_music_list :
    filename = 'STM_output/corpSTMnpy/'+corp.replace('/', '-')+'_STMall.npy'
    x = tsne.transform(np.load(filename))
