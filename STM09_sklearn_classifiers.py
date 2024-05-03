#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:11:23 2024

@author: andrewchang
"""

import numpy as np
import pandas as pd
import datetime
import sys


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



def prepData():
    # %% load STM data
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
    
    # %% load meta data
    speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
    speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv',index_col=0)
    music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
    df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)
    
    all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)

    # %% split data
    
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
    
    data_split = pd.concat([all_corp_df['10fold_labels'], pd.Series([1] * SONYC_aug_len)],
                           ignore_index=True)
    
    train_ind = data_split<8
    val_ind = data_split==8
    test_ind = data_split==9
    
    X_train = STM_all[train_ind,:]
    X_val = STM_all[val_ind,:]
    X_test = STM_all[test_ind,:]
    
    y_train = target[train_ind]
    y_val = target[val_ind]
    y_test = target[test_ind]

    return X_train, X_val, X_test, y_train, y_val, y_test

# %% linearSVC
def run_linearSVC(X_train, X_val, y_train, y_val):
    def bo_tune_linearSVC(C_power):
        params = {
            "kernel":"linear",
            "C":10**C_power,
            "probability":True,
            "class_weight":"balanced",
            "decision_function_shape":"ovr",
            "random_state":23,
            "verbose":1
            }
        clf = make_pipeline(StandardScaler(), SVC(**params))
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val), multi_class='ovr')
    
    bo_linearSVC = BayesianOptimization(
        bo_tune_linearSVC,
        pbounds={
            "C_power": (-6, 6),
            },
        random_state=23
        )
    logger = JSONLogger(path='model/sklearn_corpora_categories/linearSVM/logs_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.json')
    bo_linearSVC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_linearSVC.maximize(n_iter=200, init_points=25)

# %% rbfSVC
def run_rbfSVC(X_train, X_val, y_train, y_val):
    def bo_tune_rbfSVC(C_power, gamma_power):
        params = {
            "kernel":"rbf",
            "C":10**C_power,
            "gamma":10**gamma_power,
            "probability":True,
            "class_weight":"balanced",
            "decision_function_shape":"ovr",
            "random_state":23,
            "verbose":1
            }
        clf = make_pipeline(StandardScaler(), SVC(**params))
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val), multi_class='ovr')
    
    bo_rbfSVC = BayesianOptimization(
        bo_tune_rbfSVC,
        pbounds={
            "C_power": (-6, 6),
            "gamma_power": (-6, 0),
            },
        random_state=23
        )
    logger = JSONLogger(path='model/sklearn_corpora_categories/rbfSVM/logs_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.json')
    bo_rbfSVC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_rbfSVC.maximize(n_iter=200, init_points=25)

# %% Logistic Regression
def run_LogReg(X_train, X_val, y_train, y_val):
    def bo_tune_LogReg(C_power, l1_ratio):
        params = {
            "penalty":"elasticnet",
            "C":10**C_power,
            "l1_ratio":l1_ratio,
            "solver":"saga",
            "n_jobs":-1,
            "class_weight":"balanced",
            "multi_class":"ovr",
            "random_state":23,
            "verbose":1
            }
        clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val), multi_class='ovr')
    
    bo_LogReg = BayesianOptimization(
        bo_tune_LogReg,
        pbounds={
            "C_power": (-6, 6),
            "l1_ratio": (0, 1),
            },
        random_state=23
        )
    logger = JSONLogger(path='model/sklearn_corpora_categories/logReg/logs_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.json')
    bo_LogReg.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_LogReg.maximize(n_iter=200, init_points=25)

# %% Random Forest Classifier

def run_RFC(X_train, X_val, y_train, y_val):
    def bo_tune_RFC(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_samples):
        params = {
            "n_estimators":int(n_estimators),
            "max_depth":int(max_depth),
            "min_samples_leaf":int(min_samples_leaf),
            "min_samples_split":int(min_samples_split),
            "max_samples":max_samples,
            "n_jobs":-1,
            "class_weight":"balanced",
            "random_state":23,
            "verbose":1
            }
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(**params))
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val), multi_class='ovr')
    
    bo_RFC = BayesianOptimization(
        bo_tune_RFC,
        pbounds={
            "n_estimators": (100,1000),
            "max_depth": (3, 50),
            "min_samples_leaf": (1, 5),
            "min_samples_split": (2, 10),
            "max_samples": (10e-5,1)
            },
        random_state=23
        )
    logger = JSONLogger(path='model/sklearn_corpora_categories/randomForest/logs_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.json')
    bo_RFC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_RFC.maximize(n_iter=100, init_points=25)

# %% main
if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script.py arg1")
        sys.exit(1)
    
    clf_type = sys.argv[1]
    X_train, X_val, X_test, y_train, y_val, y_test = prepData()
    
    if clf_type=='linearSVC':
        run_linearSVC(X_train, X_val, y_train, y_val)
    elif clf_type=='rbfSVC':
        run_rbfSVC(X_train, X_val, y_train, y_val)
    elif clf_type=='LogReg':
        run_LogReg(X_train, X_val, y_train, y_val)  
    elif clf_type=='RFC':
        run_RFC(X_train, X_val, y_train, y_val)
    else:
        print("choose a classifer names: [linearSVC, rbfSVC, LogReg, RFC]")
    
    print("All Done!")