#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:38:10 2024

@author: andrewchang
"""
import pandas as pd
import glob
metaData_name_list = glob.glob('metaTables/*.csv')

def categorize_file(path):
    if 'data/musicCorp/' in path:
        return 'music'
    elif 'data/speechCorp/' in path:
        return 'speech'
    elif 'data/envCorp/' in path:
        return 'env'

for metaData_name in metaData_name_list:
    df = pd.read_csv(metaData_name,index_col=0)   
    df['corpus_type'] = df['filepath'].apply(categorize_file)
    df.to_csv(metaData_name)