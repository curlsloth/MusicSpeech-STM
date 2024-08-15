#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:25:58 2024

"""
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump
import datetime
import sys
import os 


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

corpus_env_list = [
    'SONYC', 
    'MacaulayLibrary',
]

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
    
# %% run code
perplexity = int(sys.argv[1])
if perplexity == 0:
    # PCA
    pipeline = make_pipeline(StandardScaler(),IncrementalPCA())
    pipeline.fit(STM_all)
    dump(pipeline, 'model/STM/PCA/allSTM_pca-pipeline_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.joblib') 
    
    # MDS (too big)
    # mds = MDS(
    #     n_components=2, 
    #     random_state=23, 
    #     n_jobs=-1,
    #     verbose=1,
    #     )
    # pipeline_MDS = make_pipeline(StandardScaler(), mds)
    # STM_MDS = pipeline_MDS.fit_transform(STM_all)
    # path = 'model/MDS/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # os.mkdir(path)
    # dump(pipeline_MDS, path+'/allSTM_MDS-pipeline.joblib') 
    # dump(STM_MDS, path+'/allSTM_MDS-data.joblib') 
else:
    tsne = TSNE(n_components=2, 
                random_state=23, 
                perplexity=perplexity,
                verbose=1, 
                n_jobs=-1)
    STM_tsne = tsne.fit_transform(STM_all)
    path = 'model/STM/tsne/perplexity'+str(perplexity)+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.mkdir(path)
    dump(tsne, path+'/tsne_model.joblib') 
    dump(STM_tsne, path+'/tsne_data.joblib') 
