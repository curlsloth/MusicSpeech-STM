#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:18:30 2024

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
import scipy.io

def mask_STMmatrix(ablation_params):
    x_lowcutoff = ablation_params['x_lowcutoff']
    x_highcutoff = ablation_params['x_highcutoff']
    y_lowcutoff = ablation_params['y_lowcutoff']
    y_highcutoff = ablation_params['y_highcutoff']
    
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
    
    # mask the matrix
    matrix = np.full((len(y_axis_small), len(x_axis_small)), 0)
    if x_lowcutoff is not None:
        matrix[:,np.abs(x_axis_small)<=x_lowcutoff]=1
    if x_highcutoff is not None:
        matrix[:,np.abs(x_axis_small)>x_highcutoff]=1
    if y_lowcutoff is not None:
        matrix[np.abs(y_axis_small)<=y_lowcutoff,:]=1
    if y_highcutoff is not None:
        matrix[np.abs(y_axis_small)>y_highcutoff,:]=1
    return matrix

def prepData_STM(addAug=False, ds_nontonal_speech=False, ablation_params=None):
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
    
    if addAug:
        corpus_env_list = ['SONYC', 'MacaulayLibrary', 'SONYC_augmented']
    else:
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
            if corp == 'SONYC_augmented':
                SONYC_aug_len = np.load(filename).shape[0]
        print(filename)
        
    if ablation_params is not None: # use random numbers between 0 and 1 to replace the selected STM region
        mask_matrix = mask_STMmatrix(ablation_params).flatten()
        # np.random.seed(23)
        # STM_all[:, mask_matrix==1] = np.random.rand(STM_all.shape[0], np.sum(mask_matrix))
        # STM_all[:, mask_matrix==1] = 0.0 # put the filtered out regions as 0
        STM_all = STM_all[:,mask_matrix==0] # exclude the filtered out regions (Sept 6)
        del mask_matrix
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    
    
    
    
    # add augmented enviromental sounds
    if addAug:
        target = pd.concat([all_corp_df['corpus_type'], pd.Series(['env'] * SONYC_aug_len)], 
                           ignore_index=True)
        data_split = pd.concat([all_corp_df['10fold_labels'], pd.Series([1] * SONYC_aug_len)],
                               ignore_index=True)
    else:
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
    
    y = keras.utils.to_categorical(target, num_classes=len(target.unique()))
    
    
    if ds_nontonal_speech: # whether to downsample the nontonal_speech category
        # Number of rows to sample for target == 0
        num_samples = 100000

        # Get indices of rows where target == 0
        indices_target_0 = target.index[target == 0].to_numpy()

        # Check if there are enough rows to sample
        if len(indices_target_0) < num_samples:
            raise ValueError(f"There are not enough rows with target == 0 to sample {num_samples} rows.")

        # Randomly sample indices from the rows where target == 0
        np.random.seed(23)
        sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)

        # Create a mask for the entire array, starting with selecting all rows
        mask = np.ones(len(target), dtype=bool)

        # Set the mask to False for rows where target == 0 but not in sampled_indices
        mask[indices_target_0] = False
        mask[sampled_indices] = True

        # Apply the mask to the NumPy array
        STM_all = STM_all[mask,:]
        data_split = data_split[mask]
        y = y[mask,:]
       
        
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




def prepData_VGG(ds_nontonal_speech = False):
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
        # 'CD',
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC', 'MacaulayLibrary'] # no SONYC_augmented
    
    # sort the corpora lists to make sure the order is replicable
    corpus_speech_list.sort()
    corpus_music_list.sort()
    corpus_env_list.sort()
    
    corpus_list_all = corpus_speech_list+corpus_music_list+corpus_env_list 
    
    for corp in corpus_list_all:
        filename = 'vggish_output/embeddings/'+corp.replace('/', '-')+'_vggishEmbeddings.npy'
        if 'emb_all' not in locals():
            emb_all = np.load(filename)
        else:
            emb_all = np.vstack((emb_all, np.load(filename)))
        print(filename)
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    
    
    target = all_corp_df['corpus_type']
    
    target.replace({
        'speech: non-tonal':0,
        'speech: tonal':1,
        'music: vocal':2,
        'music: non-vocal':3,
        'env: urban':4,
        'env: wildlife':5,
        },
        inplace=True)
    
    y = keras.utils.to_categorical(target, num_classes=len(target.unique()))
    
    data_split = all_corp_df['10fold_labels']
    
    
    
    
    if ds_nontonal_speech: # whether to downsample the nontonal_speech category
        # Number of rows to sample for target == 0
        num_samples = 100000

        # Get indices of rows where target == 0
        indices_target_0 = target.index[target == 0].to_numpy()

        # Check if there are enough rows to sample
        if len(indices_target_0) < num_samples:
            raise ValueError(f"There are not enough rows with target == 0 to sample {num_samples} rows.")

        # Randomly sample indices from the rows where target == 0
        np.random.seed(23)
        sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)

        # Create a mask for the entire array, starting with selecting all rows
        mask = np.ones(len(target), dtype=bool)

        # Set the mask to False for rows where target == 0 but not in sampled_indices
        mask[indices_target_0] = False
        mask[sampled_indices] = True

        # Apply the mask to the NumPy array
        emb_all = emb_all[mask,:]
        data_split = data_split[mask]
        y = y[mask,:]
        
        
        
        
    
    # % split data

    train_ind = data_split<8
    val_ind = data_split==8
    test_ind = data_split==9
    
    train_dataset = tf.data.Dataset.from_tensor_slices((emb_all[train_ind,:], y[train_ind,:]))
    val_dataset = tf.data.Dataset.from_tensor_slices((emb_all[val_ind,:], y[val_ind,:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((emb_all[test_ind,:], y[test_ind,:]))
    
    # shuffle and then batch
    batch_size = 256

    train_dataset = train_dataset.shuffle(buffer_size=sum(train_ind), seed=23).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=sum(val_ind), seed=23).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(batch_size)
    
    n_feat = emb_all.shape[1]
    n_target = len(target.unique())
    
   
    if len(target) != len(emb_all):
        print("Embedding data and meta data mismatched!")
    elif sum(train_ind)+sum(val_ind)+sum(test_ind) != len(emb_all):
        print("Data split wrong")
    else:
        print("Good to go!")
        
    return train_dataset, val_dataset, test_dataset, n_feat, n_target




def prepData_YAM(ds_nontonal_speech = False):
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
        # 'CD',
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC', 'MacaulayLibrary'] # no SONYC_augmented
    
    # sort the corpora lists to make sure the order is replicable
    corpus_speech_list.sort()
    corpus_music_list.sort()
    corpus_env_list.sort()
    
    corpus_list_all = corpus_speech_list+corpus_music_list+corpus_env_list 
    
    for corp in corpus_list_all:
        filename = 'yamnet_output/embeddings/'+corp.replace('/', '-')+'_yamnetEmbeddings.npy'
        if 'emb_all' not in locals():
            emb_all = np.load(filename)
        else:
            emb_all = np.vstack((emb_all, np.load(filename)))
        print(filename)
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    
    
    target = all_corp_df['corpus_type']
    
    target.replace({
        'speech: non-tonal':0,
        'speech: tonal':1,
        'music: vocal':2,
        'music: non-vocal':3,
        'env: urban':4,
        'env: wildlife':5,
        },
        inplace=True)
    
    y = keras.utils.to_categorical(target, num_classes=len(target.unique()))
    
    data_split = all_corp_df['10fold_labels']
    
    
    
    if ds_nontonal_speech: # whether to downsample the nontonal_speech category
        # Number of rows to sample for target == 0
        num_samples = 100000

        # Get indices of rows where target == 0
        indices_target_0 = target.index[target == 0].to_numpy()

        # Check if there are enough rows to sample
        if len(indices_target_0) < num_samples:
            raise ValueError(f"There are not enough rows with target == 0 to sample {num_samples} rows.")

        # Randomly sample indices from the rows where target == 0
        np.random.seed(23)
        sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)

        # Create a mask for the entire array, starting with selecting all rows
        mask = np.ones(len(target), dtype=bool)

        # Set the mask to False for rows where target == 0 but not in sampled_indices
        mask[indices_target_0] = False
        mask[sampled_indices] = True

        # Apply the mask to the NumPy array
        emb_all = emb_all[mask,:]
        data_split = data_split[mask]
        y = y[mask,:]
    
    

    # % split data
    train_ind = data_split<8
    val_ind = data_split==8
    test_ind = data_split==9
    
    train_dataset = tf.data.Dataset.from_tensor_slices((emb_all[train_ind,:], y[train_ind,:]))
    val_dataset = tf.data.Dataset.from_tensor_slices((emb_all[val_ind,:], y[val_ind,:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((emb_all[test_ind,:], y[test_ind,:]))
    
    # shuffle and then batch
    batch_size = 256

    train_dataset = train_dataset.shuffle(buffer_size=sum(train_ind), seed=23).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=sum(val_ind), seed=23).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(batch_size)
    
    n_feat = emb_all.shape[1]
    n_target = len(target.unique())
    
   
    if len(target) != len(emb_all):
        print("Embedding data and meta data mismatched!")
    elif sum(train_ind)+sum(val_ind)+sum(test_ind) != len(emb_all):
        print("Data split wrong")
    else:
        print("Good to go!")
        
    return train_dataset, val_dataset, test_dataset, n_feat, n_target




def prepData_melspectrogram(ds_nontonal_speech = False):
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
        # 'CD',
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC', 'MacaulayLibrary'] # no SONYC_augmented
    
    # sort the corpora lists to make sure the order is replicable
    corpus_speech_list.sort()
    corpus_music_list.sort()
    corpus_env_list.sort()
    
    corpus_list_all = corpus_speech_list+corpus_music_list+corpus_env_list 
    
    for corp in corpus_list_all:
        # filename = 'melspectrogram_output/'+corp.replace('/', '-')+'_melspectrogram.npy'
        filename = 'melspectrogram_output/'+corp.replace('/', '-')+'_melspectrogram.npy'
        if 'emb_all' not in locals():
            emb_all = np.load(filename)
        else:
            emb_all = np.vstack((emb_all, np.load(filename)))
        print(filename)
        
    # % load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
    
    
    
    target = all_corp_df['corpus_type']
    
    target.replace({
        'speech: non-tonal':0,
        'speech: tonal':1,
        'music: vocal':2,
        'music: non-vocal':3,
        'env: urban':4,
        'env: wildlife':5,
        },
        inplace=True)
    
    y = keras.utils.to_categorical(target, num_classes=len(target.unique()))
    
    data_split = all_corp_df['10fold_labels']
    
    
    
    
    if ds_nontonal_speech: # whether to downsample the nontonal_speech category
        # Number of rows to sample for target == 0
        num_samples = 100000

        # Get indices of rows where target == 0
        indices_target_0 = target.index[target == 0].to_numpy()

        # Check if there are enough rows to sample
        if len(indices_target_0) < num_samples:
            raise ValueError(f"There are not enough rows with target == 0 to sample {num_samples} rows.")

        # Randomly sample indices from the rows where target == 0
        np.random.seed(23)
        sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)

        # Create a mask for the entire array, starting with selecting all rows
        mask = np.ones(len(target), dtype=bool)

        # Set the mask to False for rows where target == 0 but not in sampled_indices
        mask[indices_target_0] = False
        mask[sampled_indices] = True

        # Apply the mask to the NumPy array
        emb_all = emb_all[mask,:]
        data_split = data_split[mask]
        y = y[mask,:]
        
        
        
        
    
    # % split data

    train_ind = data_split<8
    val_ind = data_split==8
    test_ind = data_split==9
    
    train_dataset = tf.data.Dataset.from_tensor_slices((emb_all[train_ind,:], y[train_ind,:]))
    val_dataset = tf.data.Dataset.from_tensor_slices((emb_all[val_ind,:], y[val_ind,:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((emb_all[test_ind,:], y[test_ind,:]))
    
    # shuffle and then batch
    batch_size = 256

    train_dataset = train_dataset.shuffle(buffer_size=sum(train_ind), seed=23).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=sum(val_ind), seed=23).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=sum(test_ind), seed=23).batch(batch_size)
    
    n_feat = emb_all.shape[1]
    n_target = len(target.unique())
    
   
    if len(target) != len(emb_all):
        print("Embedding data and meta data mismatched!")
    elif sum(train_ind)+sum(val_ind)+sum(test_ind) != len(emb_all):
        print("Data split wrong")
    else:
        print("Good to go!")
        
    return train_dataset, val_dataset, test_dataset, n_feat, n_target