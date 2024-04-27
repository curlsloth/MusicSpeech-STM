#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:49:01 2024

@author: andrewchang
"""

import scipy.io
import numpy as np
import time
import pandas as pd
import glob
import random

def preproSTM(data, xmin, xmax, ymin, ymax, x_ds_factor, y_ds_factor):
    # select the middle range, dB transformation, normalize to [0,1], make it as float 32
    data_small_db = 10 * np.log10(data[ymin:ymax+1:y_ds_factor, xmin:xmax+1:x_ds_factor]) # power dB
    return np.float32((data_small_db - data_small_db.min()) / (data_small_db.max() - data_small_db.min()))


def stack_STM(df):
    stm_stacked_list = []

    # hard-coded parameters. Use the code below to modify it
    # xrange = [-15, 15]
    # yrange = [0,7.2]
    xmin=190
    xmax=310
    ymin=75
    ymax=114
    # downsampling factor per axis
    x_ds_factor=1
    y_ds_factor=2

    for mat_file in df['mat_filename']:
        try:
            data = scipy.io.loadmat(mat_file)['indMS']
            if data.shape == (150, 500):
                stm_stacked_list.append(preproSTM(data, xmin, xmax, ymin, ymax, x_ds_factor, y_ds_factor).flatten())
            else:
                print('indMS size wrong: '+str(data.shape)+mat_file)
        except Exception as e:
            print("An error occurred:", e)
                
    return np.vstack(stm_stacked_list)

def generate_aug_envSTM(stm_stacked, n_aug, n_sample_max, labels):
    """
    Parameters
    ----------
    n_aug : int
        number of augmented data
    n_sample_max : int
        the number of random sampled env STM data, between [2, n_sample_max]
    """
    
    stm_stacked = stm_stacked[labels<9,:] # exclude the evaluation set
    
    stm_new_list = []
    for n_seed in range(n_aug):
        # Set the seed for reproducibility
        random.seed(n_seed)
        
        # Generate a random number between 2 and n_sample_max, which is the number of samples in this iteration
        n_sample = random.randint(2,n_sample_max)
            
        # Generate random numbers between 1 and 100
        numbers = [random.randint(0,len(stm_stacked)-1) for _ in range(n_sample)]
        
        # Generate random positive weights that sum up to 1
        weights = [random.random() for _ in range(n_sample)]
        weights = [weight / sum(weights) for weight in weights]
        
        for n in range(n_sample):
            if n == 0:
                stm_new = stm_stacked[numbers[n]]*weights[n]
            else:
                stm_new += stm_stacked[numbers[n]]*weights[n]
        stm_new = (stm_new - stm_new.min()) / (stm_new.max() - stm_new.min()) # make sure it is between 0 and 1
        stm_new_list.append(stm_new)
        
    return np.vstack(stm_new_list)

# %% generate augmented dataset of the environmental sound

df = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)   

n_aug = 100000 # number of augmented incidence
n_sample_max = 5 # the number of random sampled env STM data, between [2, n_sample_max]

stm_stacked_aug = generate_aug_envSTM(stack_STM(df), n_aug, n_sample_max, labels=df['10fold_labels'])
np.save('STM_output/corpSTMnpy/SONYC_augmented_STMall.npy', stm_stacked_aug)

# %% stack all the output
metaData_name_list = glob.glob('metaTables/*.csv')
for metaData_name in metaData_name_list:
    t_start = time.time()
    print(metaData_name)
    df = pd.read_csv(metaData_name,index_col=0)   
    corpus_name = metaData_name[20:-4]
    np.save('STM_output/corpSTMnpy/'+corpus_name+'_STMall.npy', stack_STM(df))
    print('time elapsed: '+str(time.time()-t_start)+' seconds')
    

# %% plot 2D STM
# import matplotlib.pyplot as plt
# plt.pcolormesh(x_axis_small, y_axis_small, data_proc)  # You can choose any colormap you prefer
# plt.colorbar()  # Add a colorbar for reference
# plt.show()

# %% load one survey data to set the axis parameters
# df = pd.read_csv('metaTables/metaData_BibleTTS-hausa.csv',index_col=0)
# survey_file = df['mat_filename'][0].replace('MATs','Survey').replace('mat_wl4','params').replace('MS2024','Params')
# survey_data = scipy.io.loadmat(survey_file)
# x_axis = survey_data['Params'][0]['x_axis'][0][0]
# y_axis = survey_data['Params'][0]['y_axis'][0][0]

# xrange = [-15, 15]
# yrange = [0,7.2]

# x_ds_factor = 1
# y_ds_factor = 2

# xmin = np.argmin(np.abs(x_axis - xrange[0]))
# xmax = np.argmin(np.abs(x_axis - xrange[1]))
# ymin = np.argmin(np.abs(y_axis - yrange[0]))
# ymax = np.argmin(np.abs(y_axis - yrange[1]))
# x_axis_small = x_axis[xmin:xmax+1:x_ds_factor]
# y_axis_small = y_axis[ymin:ymax+1:y_ds_factor]