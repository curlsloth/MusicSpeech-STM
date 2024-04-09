#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:35:14 2024

@author: andrewchang

Install demucs: python3 -m pip install -U git+https://github.com/adefossez/demucs#egg=demucs
"""

import demucs.api
import soundfile as sf
import torchaudio
import torch
import scipy.io as sio
import tensorflow as tf
import scipy
import numpy as np
import csv
import pandas as pd
import time
import sys

# import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile





# modified from here: https://www.tensorflow.org/hub/tutorials/yamnet
# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names



# verify and convert a loaded audio is on the proper sample_rate (16K), otherwise it would affect the model's results.
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def estimate_voice(df_meta, n_row):
    # st = time.time()
    path = 'yamnet/tensorFlow2/yamnet/1'
    model = tf.saved_model.load(path, tags=None, options=None)
    
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)
    
    filename = df_meta['filepath'].iloc[n_row]
    frame_offset = df_meta['startPoint'].iloc[n_row]-1
    num_frame = df_meta['endPoint'].iloc[n_row]-frame_offset+1
    y_tensor, sr = torchaudio.load(filename, frame_offset=frame_offset, num_frames=num_frame)
    if y_tensor.shape[0]==1: # if the audio is mono, duplicate the channel
        y_tensor = torch.stack((y_tensor, y_tensor), dim=0)
    separator = demucs.api.Separator()
    origin, separated = separator.separate_tensor(wav=y_tensor, sr=sr) # demucs.api.Separator() by default will make sure the sampling rate is at 44.1 kHz
    _, waveform = ensure_sample_rate(44100, np.array(separated['vocals'].mean(axis=0))) # convert to sr=16000
    scores, _, _ = model(np.array(separated['vocals'].mean(axis=0))) # use YAMNET to score the audio waveform
    scores_np = scores.numpy()
    class_timeseries = scores_np.argmax(axis=1) # the max likelihood category per time-frame
    voice = np.any((class_timeseries >= 0) & (class_timeseries <= 35)) # whether the label of any time-frame belongs to any voice labels
    # infered_class = class_names[scores_np.mean(axis=0).argmax()]
    # voice = infered_class in class_names[0:35] # whether 
    # # get the execution time
    # et = time.time()
    # print('Execution time:', et - st, 'seconds')
    return voice

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py arg1 arg2")
        sys.exit(1)

    # Extract command-line arguments
    df_meta = sys.argv[1]
    init_num = sys.argv[2]*100

    # Call your function or perform any desired operations
    for n in range(init_num, init_num+100):
        estimate_voice(df_meta, n)
        
        ## how to save the file? How to show the outcome?
        
    print("Result:", result)