#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:16:30 2024

@author: andrewchang
"""

n_row=678
path = 'metaTables'
df_meta_filename = 'metaData_ismir04_genre.csv'
df_meta = pd.read_csv(os.path.join(path,df_meta_filename), index_col=0)
savefile_path = os.path.join(path, 'vocal_music_demucs_16k', df_meta_filename[:-4])
savefile_name = os.path.join(savefile_path, 'row'+str(n_row)+'.csv')

print(df_meta['filepath'].iloc[n_row])

voice, filename = estimate_voice(df_meta, n_row)

with open(savefile_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([filename, voice])
    
print(savefile_name)
os.path.isfile(savefile_name)