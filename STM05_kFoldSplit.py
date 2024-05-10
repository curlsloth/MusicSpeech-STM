#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:19:01 2024

@author: andrewchang
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


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

def SGKF(n_splits,X,y,groups):
    fold_labels_df = pd.DataFrame({'fold_labels': np.nan}, index=range(len(y)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=23)
    
    for i, (train_index, test_index) in enumerate(sgkf.split(X, y, list(groups))):
        fold_labels_df.loc[test_index,'fold_labels'] = i
    
    return fold_labels_df

def GKF(n_splits,X,y,groups):
    fold_labels_df = pd.DataFrame({'fold_labels': np.nan}, index=range(len(y)))
    gkf = GroupKFold(n_splits=n_splits)
    
    for i, (train_index, test_index) in enumerate(gkf.split(X, y, list(groups))):
        fold_labels_df.loc[test_index,'fold_labels'] = i
    
    return fold_labels_df


# %% speech data cleaning

speech_corp_list = []
for corp in corpus_speech_list:
    metafile = 'metaTables/metaData_'+corp.replace('/', '-')+'.csv'
    speech_corp_list.append(pd.read_csv(metafile,index_col=0))
speech_corp_df = pd.concat(speech_corp_list, ignore_index=True)

speech_corp_df.drop(columns='genre', inplace=True)

speech_corp_df = unify_gender(speech_corp_df)

gender_count = speech_corp_df['gender'].value_counts()

# tone languages, # reference : https://wals.info/feature/13A?s=20&z3=3000&z2=2999&z1=2998&tg_format=map&v1=cfff&v2=cf6f&v3=cd00#2/19.3/152.8
# at the moment, only consider complex tone languages
complex_tone_lang_list = [
    'Cantonese',
    'Mandarin-China', 
    'Mandarin-Taiwan',
    'Taiwanese',
    'Thai',
    'Vietnamese',
    'Yoruba'
    ]
# simple_tone_lang_list = [
#     'Ewe',
#     'Hadza',
#     'Hausa',
#     'Igbo',
#     'Japanese',
#     'Latvian',
#     ]
# tone_lang_list = complex_tone_lang_list+simple_tone_lang_list

toneLang_mask = speech_corp_df['LangOrInstru'].isin(complex_tone_lang_list)
nontoneLang_mask = ~speech_corp_df['LangOrInstru'].isin(complex_tone_lang_list)
toneLang_mask = toneLang_mask.reset_index(drop=True)
nontoneLang_mask = nontoneLang_mask.reset_index(drop=True)
speech_corp_df.loc[toneLang_mask, 'corpus_type'] = 'speech: tonal'
speech_corp_df.loc[nontoneLang_mask, 'corpus_type'] = 'speech: non-tonal'


speech_corp_df['10fold_labels'] = SGKF(n_splits = 10,
                                       X=speech_corp_df,
                                       y=speech_corp_df['corpus_type'],
                                       groups=speech_corp_df['speaker/artist'])


print("Total length of speech recordings: "+ str(round(speech_corp_df['totalLengCur'].sum()/(60*60),2))+" hours")
print("Number of speech recordings: "+ str(len(speech_corp_df)))
print("Number of unique speakers: "+ str(len(speech_corp_df['speaker/artist'].unique())))

print("Number of tonal languages: "+ str(sum(speech_corp_df['corpus_type']=='speech: tonal')))
print("Number of non tonal languages: "+ str(sum(speech_corp_df['corpus_type']=='speech: non-tonal')))

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

song_mask = (music_corp_df['corpus_type'] == 'music') & (music_corp_df['VoiOrNot'].isin([True, 1]))
song_mask = song_mask.reset_index(drop=True)
music_corp_df.loc[song_mask, 'corpus_type'] = 'music: vocal'
music_corp_df.loc[~song_mask, 'corpus_type'] = 'music: non-vocal'

music_corp_df['10fold_labels'] = SGKF(n_splits = 10,
                                      X=music_corp_df,
                                      y=music_corp_df['corpus_type'],
                                      groups=music_corp_df['speaker/artist'])

print("Total length of music recordings: "+ str(round(music_corp_df['totalLengCur'].sum()/(60*60),2))+" hours")
print("Number of music recordings: "+ str(len(music_corp_df)))
print("Number of unique musicians/ensembles: "+ str(len(music_corp_df['speaker/artist'].unique())))
print("Number of vocal music: "+ str(sum(music_corp_df['corpus_type']=='music: vocal')))
print("Number of non-vocal music: "+ str(sum(music_corp_df['corpus_type']=='music: non-vocal')))

print("Number of genres: "+str(len(genres)))
print(genres)

# %% env sounds

df_SONYC = pd.read_csv('metaTables/metaData_SONYC.csv',index_col=0).reset_index(drop=True)

df_SONYC['10fold_labels'] = GKF(n_splits = 10,
                                 X=df_SONYC,
                                 y=df_SONYC['corpus_type'],
                                 groups=df_SONYC['speaker/artist'])

print("Total length of environmental recordings: "+ str(round(df_SONYC['totalLengCur'].sum()/(60*60),2))+" hours")
print("Number of environmental recordings: "+ str(len(df_SONYC)))

# %% save split files

save_split_files = False

if save_split_files:
    speech_row_split = round(len(speech_corp_df)/2)
    speech_corp_df[['mat_filename','corpus_type','10fold_labels']][:speech_row_split].to_csv('train_test_split/speech1_10folds_speakerGroupFold.csv')
    speech_corp_df[['mat_filename','corpus_type','10fold_labels']][speech_row_split:].to_csv('train_test_split/speech2_10folds_speakerGroupFold.csv')

    music_corp_df[['mat_filename','corpus_type','10fold_labels']].to_csv('train_test_split/music_10folds_speakerGroupFold.csv')
    df_SONYC[['mat_filename','corpus_type','10fold_labels']].to_csv('train_test_split/env_10folds_speakerGroupFold.csv')

