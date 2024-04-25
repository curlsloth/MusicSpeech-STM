#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:19:01 2024

@author: andrewchang
"""

import numpy as np
import pandas as pd
from joblib import dump, load
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


# %% corpus lists

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


# %% shared functions
def unify_gender(df):
    df['gender'] = df['gender'].replace({
        'm':'male',
        'M':'male',
        'f':'female',
        'F':'female',
        'other':np.nan,
        'O':np.nan,
        })
    return df

# %% speech data cleaning

speech_corp_list = []
for corp in corpus_speech_list:
    metafile = 'metaTables/metaData_'+corp.replace('/', '-')+'.csv'
    speech_corp_list.append(pd.read_csv(metafile,index_col=0))
speech_corp_df = pd.concat(speech_corp_list, ignore_index=True)

speech_corp_df = unify_gender(speech_corp_df)

gender_count = speech_corp_df['gender'].value_counts()



print("Total length of speech recordings: "+ str(round(speech_corp_df['totalLengCur'].sum()/(60*60),2))+" hours")
print("Number of speech recordings: "+ str(len(speech_corp_df)))
print("Number of unique speakers: "+ str(len(speech_corp_df['speaker/artist'].unique())))

print("Male ratio: "+str(round(100*gender_count['male']/len(speech_corp_df), 2))+"%")
print("Female ratio: "+str(round(100*gender_count['female']/len(speech_corp_df), 2))+"%")
print("NaN ratio: " + str(round(100*sum(speech_corp_df['gender'].isna())/len(speech_corp_df),2))+"%")

languages = speech_corp_df['LangOrInstru'].unique()
languages.sort()    
print("Number of languages: "+str(len(languages)))
print(languages)

# %% music data cleaning
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

music_corp_df = unify_gender(music_corp_df)

print("Total length of music recordings: "+ str(round(music_corp_df['totalLengCur'].sum()/(60*60),2))+" hours")
print("Number of music recordings: "+ str(len(music_corp_df)))
print("Number of unique musicians/ensembles: "+ str(len(music_corp_df['speaker/artist'].unique())))


print("Number of genres: "+str(len(genres)))
print(genres)

# %% env sounds

df_SONYC = pd.read_csv('metaTables/metaData_SONYC.csv',index_col=0)
print("Total length of environmental recordings: "+ str(round(df_SONYC['totalLengCur'].sum()/(60*60),2))+" hours")
print("Number of environmental recordings: "+ str(len(df_SONYC)))



all_corp_df = pd.concat([speech_corp_df, music_corp_df, df_SONYC])

# %% tone languages
# reference : https://wals.info/feature/13A?s=20&z3=3000&z2=2999&z1=2998&tg_format=map&v1=cfff&v2=cf6f&v3=cd00#2/19.3/152.8
complex_tone_lang_list = [
    'Cantonese',
    'Mandarin-China', 
    'Mandarin-Taiwan',
    'Taiwanese',
    'Thai',
    'Vietnamese',
    'Yoruba'
    ]
simple_tone_lang_list = [
    'Ewe',
    'Hadza',
    'Hausa',
    'Igbo',
    'Japanese',
    'Latvian',
    ]

tone_lang_list = complex_tone_lang_list+simple_tone_lang_list

# %% plot PCA

pipeline = load('model/allSTM_pca-pipeline_2024-04-22_08-43.joblib')
pca = pipeline['incrementalpca']


plt.plot(list(range(1,len(pca.explained_variance_ratio_[:5000])+1)), np.cumsum(pca.explained_variance_ratio_[:5000]))
plt.show()

# %% load t-SNE and preprocessing

tsne_folder = 'model/tsne/perplexity400_2024-04-24_18-02/'

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

toneLang_mask = (all_corp_df['corpus_type'] == 'speech') & (all_corp_df['LangOrInstru'].isin(tone_lang_list))
nontoneLang_mask = (all_corp_df['corpus_type'] == 'speech') & (~all_corp_df['LangOrInstru'].isin(tone_lang_list))
toneLang_mask = pd.concat([toneLang_mask, pd.Series([False]* (len(tsne.embedding_) - len(all_corp_df)))])
nontoneLang_mask = pd.concat([nontoneLang_mask, pd.Series([False]* (len(tsne.embedding_) - len(all_corp_df)))])
toneLang_mask = toneLang_mask.reset_index(drop=True)
nontoneLang_mask = nontoneLang_mask.reset_index(drop=True)
df_tsne.loc[toneLang_mask, 'category'] = 'speech: tonal'
df_tsne.loc[nontoneLang_mask, 'category'] = 'speech: non-tonal'

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