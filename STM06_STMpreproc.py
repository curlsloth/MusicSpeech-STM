#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:49:01 2024

@author: andrewchang
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat_data = scipy.io.loadmat('STM_output/MATs/music_mat_wl4_Albouy2020Science/English_S01_M01_MS2024.mat')
survey_data = scipy.io.loadmat('STM_output/Survey/music_params_CD/01 Andante-Adagio-Allegretto_Params.mat')

data = mat_data['indMS']

x_axis = survey_data['Params'][0]['x_axis'][0][0]
y_axis = survey_data['Params'][0]['y_axis'][0][0]

xmin = np.argmin(np.abs(x_axis - -30))
xmax = np.argmin(np.abs(x_axis - 30))
ymin = np.argmin(np.abs(y_axis - 0))
ymax = np.argmin(np.abs(y_axis - 9))


# x_neg30_ind = 130 # -30 Hz
# x_pos30_ind = 371 # 30 Hz
# y_0_ind = 75 # 0 cyc/oct
# y_pos9_ind = 124 # 0 cyc/oct

x_axis_small = x_axis[xmin:xmax+1]
y_axis_small = y_axis[ymin:ymax+1]

def preproSTM(data, xmin, xmax, ymin, ymax):
    data_small = data[ymin:ymax+1, xmin:xmax+1]
    data_small_db = 10 * np.log10(data_small) # power dB
    data_small_db_normalized = (data_small_db - data_small_db.min()) / (data_small_db.max() - data_small_db.min())
    return data_small_db_normalized


plt.pcolormesh(x_axis_small, y_axis_small, preproSTM(data, xmin, xmax, ymin, ymax))  # You can choose any colormap you prefer
plt.colorbar()  # Add a colorbar for reference
plt.show()


