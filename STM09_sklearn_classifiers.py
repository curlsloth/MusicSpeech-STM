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
import os
import json
from joblib import dump

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.kernel_approximation import RBFSampler

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


# %% prepData
def prepData():
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
    
    # %% load STM data
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

# %% SGDClassifier (linear SVC)

def params_SGDClinearSVC(alpha_power):
    params = {
        "loss":"hinge",
        "penalty":"l2",
        "alpha":10**alpha_power,
        "shuffle":True,
        "n_jobs":-1,
        "class_weight":"balanced",
        "random_state":23,
        "verbose":1
        }
    return params

def run_SGDClinearSVC(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_SGDClinearSVC(alpha_power):
        params = params_SGDClinearSVC(alpha_power)
        clf = make_pipeline(StandardScaler(), SGDClassifier(**params))
        clf.fit(X_train, y_train)
        y_val_encoded = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(y_val))
        return roc_auc_score(y_val_encoded, clf.decision_function(X_val), multi_class='ovr')
    
    bo_SGDClinearSVC = BayesianOptimization(
        bo_tune_SGDClinearSVC,
        pbounds={
            "alpha_power": (-6, 6),
            },
        random_state=23
        )
    jogger_path = 'model/sklearn_corpora_categories/SGDClinearSVM/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_SGDClinearSVC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_SGDClinearSVC.maximize(n_iter=200, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters
    
    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_SGDClinearSVC(best_param_dict['alpha_power'])
    clf = make_pipeline(StandardScaler(), SGDClassifier(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=False)

# %% SGDClassifier (rbf SVC)

def params_SGDCrbfSVC(alpha_power):
    params = {
        "loss":"hinge",
        "penalty":"l2",
        "alpha":10**alpha_power,
        "shuffle":True,
        "n_jobs":-1,
        "class_weight":"balanced",
        "random_state":23,
        "verbose":1
        }
    return params

def params_RBFSampler(gamma_power, n_components):
    params_rbf = {
        "gamma":10**gamma_power,
        "n_components":int(n_components),
        "random_state":23,
        }
    return params_rbf

def run_SGDCrbfSVC(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_SGDCrbfSVC(alpha_power, gamma_power, n_components):
        params = params_SGDClinearSVC(alpha_power)
        params_rbf = params_RBFSampler(gamma_power, n_components)
        clf = make_pipeline(StandardScaler(), RBFSampler(**params_rbf), SGDClassifier(**params))
        clf.fit(X_train, y_train)
        y_val_encoded = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(y_val))
        return roc_auc_score(y_val_encoded, clf.decision_function(X_val), multi_class='ovr')
    
    bo_SGDCrbfSVC = BayesianOptimization(
        bo_tune_SGDCrbfSVC,
        pbounds={
            "alpha_power": (-6, 6),
            "gamma_power": (-3, 3),
            "n_components": (100, 2000)
            },
        random_state=23
        )
    jogger_path = 'model/sklearn_corpora_categories/SGDCrbfSVM/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_SGDCrbfSVC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_SGDCrbfSVC.maximize(n_iter=200, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters
    
    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_SGDCrbfSVC(best_param_dict['alpha_power'])
    params_rbf = params_RBFSampler(best_param_dict['gamma_power'], best_param_dict['n_components'])
    clf = make_pipeline(StandardScaler(), RBFSampler(**params_rbf), SGDClassifier(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=False)

# %% SGDClassifier (Logistic Regression)

def params_SGDClogReg(alpha_power, l1_ratio):
    params = {
        "loss":"log_loss",
        "penalty":"elasticnet",
        "l1_ratio":l1_ratio,
        "alpha":10**alpha_power,
        "shuffle":True,
        "n_jobs":-1,
        "class_weight":"balanced",
        "random_state":23,
        "verbose":1
        }
    return params

def run_SGDClogReg(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_SGDClogReg(alpha_power, l1_ratio):
        params = params_SGDClogReg(alpha_power, l1_ratio)
        clf = make_pipeline(StandardScaler(), SGDClassifier(**params))
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val), multi_class='ovr')
    
    bo_SGDClogReg = BayesianOptimization(
        bo_tune_SGDClogReg,
        pbounds={
            "alpha_power": (-6, 6),
            "l1_ratio": (0, 1)
            },
        random_state=23
        )
    jogger_path = 'model/sklearn_corpora_categories/SGDClogReg/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_SGDClogReg.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_SGDClogReg.maximize(n_iter=200, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters
    
    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_SGDClogReg(best_param_dict['alpha_power'], best_param_dict['l1_ratio'])
    clf = make_pipeline(StandardScaler(), SGDClassifier(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=True)

# %% linearSVC

def params_linearSVC(C_power):
    params = {
        "kernel":"linear",
        "C":10**C_power,
        "probability":True,
        "class_weight":"balanced",
        "decision_function_shape":"ovr",
        "random_state":23,
        "verbose":1
        }
    return params

def run_linearSVC(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_linearSVC(C_power):
        params = params_linearSVC(C_power)
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
    jogger_path = 'model/sklearn_corpora_categories/linearSVM/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_linearSVC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_linearSVC.maximize(n_iter=200, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters
    
    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_linearSVC(best_param_dict['C_power'])
    clf = make_pipeline(StandardScaler(), SVC(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=True)


# %% rbfSVC

def params_rbfSVC(C_power, gamma_power):
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
    return params

def run_rbfSVC(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_rbfSVC(C_power, gamma_power):
        params = params_rbfSVC(C_power, gamma_power)
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
    jogger_path = 'model/sklearn_corpora_categories/rbfSVM/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_rbfSVC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_rbfSVC.maximize(n_iter=200, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters

    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_rbfSVC(best_param_dict['C_power'], best_param_dict['gamma_power'])
    clf = make_pipeline(StandardScaler(), SVC(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=True)

# %% Logistic Regression

def params_LogReg(C_power, l1_ratio):
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
    return params

def run_LogReg(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_LogReg(C_power, l1_ratio):
        params = params_LogReg(C_power, l1_ratio)
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
    jogger_path = 'model/sklearn_corpora_categories/logReg/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_LogReg.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_LogReg.maximize(n_iter=200, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters

    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_LogReg(best_param_dict['C_power'], best_param_dict['l1_ratio'])
    clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=True)

# %% Random Forest Classifier

def params_RFC(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_samples):
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
    return params

def run_RFC(X_train, X_val, X_test, y_train, y_val, y_test):
    def bo_tune_RFC(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_samples):
        params = params_RFC(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_samples)
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
            "max_samples": (1e-4,1)
            },
        random_state=23
        )
    jogger_path = 'model/sklearn_corpora_categories/randomForest/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'/'
    os.mkdir(jogger_path)
    jogger_file=jogger_path+'logs.json'
    logger = JSONLogger(path=jogger_file)
    bo_RFC.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo_RFC.maximize(n_iter=100, init_points=25)
    best_param_dict = get_best_params(jogger_file) # this is a dict of the best parameters
    
    # fit the model again with best parameters and combined X_train X_val data
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    params = params_RFC(best_param_dict['n_estimators'], 
                        best_param_dict['max_depth'], 
                        best_param_dict['min_samples_leaf'], 
                        best_param_dict['min_samples_split'], 
                        best_param_dict['max_samples'])
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(**params))
    clf.fit(X, y)
    dump(clf, jogger_path+'clf.joblib')
    
    # print test performances
    print_performances(clf, X_test, y_test, use_prob=True)

# %% return the best parameters
def get_best_params(json_path):
    list_bayesOpt = [json_path]
    opt_best_df = pd.DataFrame()
    for jsonFile in list_bayesOpt:
        with open(jsonFile) as f:
            optList = []
            for jsonObj in f:
                optDict = json.loads(jsonObj)
                optList.append(optDict)
            
            opt_df = pd.DataFrame(optList)
            opt_df = pd.concat([opt_df.drop(['params','datetime'], axis=1), opt_df['params'].apply(pd.Series), opt_df['datetime'].apply(pd.Series)], axis=1)
            opt_best_df = pd.concat([opt_best_df,opt_df.sort_values('target',ascending=False).iloc[:10]])
    
    opt_best_df = opt_best_df.sort_values(['target', 'delta'],ascending=[False, True]) # Highest AUC, lowest training time
    print(opt_best_df)
    return dict(opt_best_df.iloc[0])

# %% print test performances
def print_performances(clf, X_test, y_test, use_prob):
    y_test_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred, normalize='true')
    if use_prob==True:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    else:
        y_test_encoded = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(y_val))
        auc = roc_auc_score(y_test_encoded, clf.decision_function(X_test), multi_class='ovr')
    print(classification_report(y_test, y_test_pred))
    print("Confusion matrix")
    print(cm)
    print("ROC AUC: "+str(auc))


# %% main
if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script.py arg1")
        sys.exit(1)
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepData()

    clf_type = sys.argv[1]
    
    if clf_type=='0':
        run_linearSVC(X_train, X_val, X_test, y_train, y_val, y_test)
    elif clf_type=='1':
        run_rbfSVC(X_train, X_val, X_test, y_train, y_val, y_test)
    elif clf_type=='2':
        run_LogReg(X_train, X_val, X_test, y_train, y_val, y_test) 
    elif clf_type=='3':
        run_RFC(X_train, X_val, X_test, y_train, y_val, y_test)
    elif clf_type=='4':
        run_SGDClinearSVC(X_train, X_val, X_test, y_train, y_val, y_test)
    elif clf_type=='5':
        run_SGDClogReg(X_train, X_val, X_test, y_train, y_val, y_test)
    elif clf_type=='6':
        run_SGDCrbfSVC(X_train, X_val, X_test, y_train, y_val, y_test)
    else:
        print("choose a classifer: {0: linearSVC, 1: rbfSVC, 2: LogReg, 3: RFC, 4: SGDClinearSVC, 5: SGDClogReg, 6: SGDCrbfSVC]")

    print("All Done!")
    