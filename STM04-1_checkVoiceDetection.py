#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:39:07 2024

@author: andrewchang
"""

# %% import modules and set corpus name

import os
import datetime
import pandas as pd

# %% functions
# check slurm output

def check_slurm(corpus_name, start_time, end_time):
    # Directory containing the .out files
    directory = 'HPC_slurm/STM04/'+corpus_name
    
    # Function to check if the string is present in a file
    def check_string_in_file(file_path, target_string):
        with open(file_path, 'r') as file:
            content = file.read()
            if target_string in content:
                return True
            else:
                return False
    
    # Target string to search for
    target_string = "Demucs voice recognition done!"
    
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
    
    # Print files that don't contain the target string
    if files_without_string:
        print("Files generated between "+str(start_time)+" and "+str(end_time)+" without the string 'Demucs voice recognition done!':")
        for file in files_without_string:
            print(file)
    elif len(filtered_files)==0:
        print("No files generated between "+str(start_time)+" and "+str(end_time))
    else:
        print("All files generated between "+str(start_time)+" and "+str(end_time)+" contain the string 'Demucs voice recognition done!'")
    
    
    
# update the original meta file
def update_meta_file(corpus_name, reset_demucs=False):
    corpus = 'metaData_'+corpus_name
    demucs_output_directory = 'metaTables/vocal_music_demucs_16k/'
    demucs_output_file_list = os.listdir(os.path.join(demucs_output_directory,corpus))
    
    file_list = []
    for file in demucs_output_file_list:
        file_path = os.path.join(demucs_output_directory, corpus, file)
        file_list.append(pd.read_csv(file_path, names = ['filepath', 'demucs_voice']))
    
    df_demucs_total = pd.concat(file_list, ignore_index=True)
    df_meta = pd.read_csv('metaTables/'+corpus+'.csv',index_col=0)
    if reset_demucs:
        df_meta.drop(columns=['demucs_voice'], inplace=True)
        
    if 'demucs_voice' not in df_meta.columns:
        if (len(df_meta) == len(df_demucs_total)) and (set(df_meta['filepath']) == set(df_demucs_total['filepath'])):
            df_metaData_demucs = df_meta.merge(df_demucs_total)
            df_metaData_demucs.to_csv('metaTables/'+corpus+'.csv')
            print('metaTables/'+corpus+'.csv file updated with demucs voice data')
        else:
            print("df_meta: "+str(len(df_meta))+" rows")
            print("df_demucs_total: "+str(len(df_demucs_total))+" rows")
            print(sum(df_demucs_total.duplicated()))
            missing = df_meta[~df_meta['filepath'].isin(df_demucs_total['filepath'])]['filepath'].tolist()
            print(missing)


# %% run the functions

corpus_name = 'fma_large'
# corpus_name = 'ismir04_genre'
# corpus_name = 'MagnaTagATune'
# corpus_name = 'MTG-Jamendo'

start_time = datetime.datetime(2024, 5, 10, 0, 0)
end_time = datetime.datetime(2024, 5, 10, 23, 0)
check_slurm(corpus_name, start_time, end_time)

change_meta_file = True
if change_meta_file:
    update_meta_file(corpus_name, reset_demucs=False)