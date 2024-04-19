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


def preproSTM(data, xmin, xmax, ymin, ymax):
    # select the middle range, dB transformation, normalize to [0,1], make it as float 32
    data_small_db = 10 * np.log10(data[ymin:ymax+1, xmin:xmax+1]) # power dB
    return np.float32((data_small_db - data_small_db.min()) / (data_small_db.max() - data_small_db.min()))


def stack_STM(df):
    stm_stacked_list = []

    # hard-coded parameters. Use the code below to modify it
    # xrange = [-20, 20]
    # yrange = [0,7]
    xmin=170
    xmax=330
    ymin=75
    ymax=112

    for mat_file in df['mat_filename']:
        data = scipy.io.loadmat(mat_file)['indMS']
        if data.shape == (150, 500):
            stm_stacked_list.append(preproSTM(data, xmin, xmax, ymin, ymax).flatten())
        else:
            print('indMS size wrong: '+str(data.shape)+mat_file)
    return np.vstack(stm_stacked_list)

def categorize_file(path):
    if 'data/musicCorp/' in path:
        return 'music'
    elif 'data/speechCorp/' in path:
        return 'speech'
    elif 'data/envCorp/' in path:
        return 'env'

metaData_name_list = glob.glob('metaTables/*.csv')
for metaData_name in metaData_name_list:
    t_start = time.time()
    print(metaData_name)
    df = pd.read_csv(metaData_name,index_col=0)   
    df['corpus_type'] = df['filepath'].apply(categorize_file)
    stm_stacked = stack_STM(df)
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

# xrange = [-20, 20]
# yrange = [0,7]

# xmin = np.argmin(np.abs(x_axis - xrange[0]))
# xmax = np.argmin(np.abs(x_axis - xrange[1]))
# ymin = np.argmin(np.abs(y_axis - yrange[0]))
# ymax = np.argmin(np.abs(y_axis - yrange[1]))
# x_axis_small = x_axis[xmin:xmax+1]
# y_axis_small = y_axis[ymin:ymax+1]