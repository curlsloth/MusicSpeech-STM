#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:13:11 2024

@author: andrewchang
"""

st = time.time()
waveform1 , sr1 = librosa.load(filename, sr=None, mono=True)
waveform1 = waveform1[frame_offset:frame_end]
et = time.time()
print('Execution time:', et - st, 'seconds')


st = time.time()
waveform2 , sr2 = sf.read(filename, frames=frame_end-frame_offset, start=frame_offset, stop=None, always_2d=True)
waveform2=waveform2.mean(axis=1)
et = time.time()
print('Execution time:', et - st, 'seconds')



# %% import modules and set corpus name

import os
import datetime
import pandas as pd

# %% functions
# check slurm output

def check_slurm(start_time, end_time):
    # Directory containing the .out files
    directory = 'HPC_slurm/STM10'
    
    # Function to check if the string is present in a file
    def check_string_in_file(file_path, target_string):
        with open(file_path, 'r') as file:
            content = file.read()
            if target_string in content:
                return True
            else:
                return False
    
    # Target string to search for
    target_string = "Execution time:"
    error_string = "***** ERROR in n_row="
    
    # List all files in the directory
    file_list = os.listdir(directory)
    
    # Filter out only the .out files
    out_files = [file for file in file_list if file.endswith('.out')]
    
    # Filter files within the specified time range
    filtered_files = []
    for file in out_files:
        file_path = os.path.join(directory, file)
        file_creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if start_time <= file_creation_time <= end_time:
            filtered_files.append(file)
    
    # Iterate over each filtered .out file and check for the target string
    files_without_string = []
    for file in filtered_files:
        file_path = os.path.join(directory, file)
        if not check_string_in_file(file_path, target_string):
            files_without_string.append(file)
        elif check_string_in_file(file_path, error_string):
            files_without_string.append(file)
    
    # Print files that don't contain the target string
    if files_without_string:
        print("Files generated between "+str(start_time)+" and "+str(end_time)+" has error:")
        for file in files_without_string:
            print(file)
    elif len(filtered_files)==0:
        print("No files generated between "+str(start_time)+" and "+str(end_time))
    else:
        print("All files generated between "+str(start_time)+" and "+str(end_time)+" contain the string 'Demucs voice recognition done!'")
    
start_time = datetime.datetime(2024, 5, 15, 0, 0)
end_time = datetime.datetime(2024, 5, 15, 23, 0)
check_slurm(start_time, end_time)