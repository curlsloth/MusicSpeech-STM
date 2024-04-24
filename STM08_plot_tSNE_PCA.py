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
from joblib import dump, load
import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %% corpus data cleaning

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

corpus_env_list = ['SONYC', 'SONYC_augmented']


# sort the corpora lists to make sure the order is replicable
corpus_speech_list.sort()
corpus_music_list.sort()
corpus_env_list.sort()

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

df_SONYC = pd.read_csv('metaTables/metaData_SONYC.csv',index_col=0)

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


all_corp_df = pd.concat([speech_corp_df, music_corp_df, df_SONYC])

# %% plot PCA

pipeline = load('model/allSTM_pca-pipeline_2024-04-22_08-43.joblib')
pca = pipeline['incrementalpca']


plt.plot(list(range(1,len(pca.explained_variance_ratio_[:5000])+1)), np.cumsum(pca.explained_variance_ratio_[:5000]))
plt.show()

# %% load t-SNE and preprocessing

tsne_folder = 'model/tsne/perplexity200_2024-04-24_07-23/'

tsne = load(tsne_folder+'tsne_model.joblib')
df_tsne = pd.DataFrame()
df_tsne['tSNE-1'] = tsne.embedding_[:,0]
df_tsne['tSNE-2'] = tsne.embedding_[:,1]
df_tsne['category'] = pd.concat([all_corp_df['corpus_type'], 
                                 pd.Series(['env_aug'] * (len(tsne.embedding_) - len(all_corp_df)))], ignore_index=True)
song_mask = (all_corp_df['corpus_type'] == 'music') & (all_corp_df['VoiOrNot'].isin([True, 1]))
song_mask = pd.concat([song_mask, pd.Series([False]* (len(tsne.embedding_) - len(all_corp_df)))])
song_mask = song_mask.reset_index(drop=True)
df_tsne.loc[song_mask, 'category'] = 'song'


# %% plot kernal 
df_tsne_filtered = df_tsne[(df_tsne['tSNE-1'].between(-60,60)) & (df_tsne['tSNE-2'].between(-60,60))]

with plt.style.context('seaborn-v0_8-notebook'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    
    kde = sns.kdeplot(
        x="tSNE-1", 
        y="tSNE-2",
        data=df_tsne_filtered,
        ax=ax,
        hue="category", 
        fill=False,
        levels=10,
        alpha=0.5,
        threshold=0
        # bw_adjust=0.1,
    )
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    plt.show()
    fig.savefig(tsne_folder+'kdeplot_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.png')