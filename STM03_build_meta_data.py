#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:38:13 2024

@author: andrewchang
"""

# %% import libraries
import scipy.io as sio
import pandas as pd
import numpy as np
import glob
import csv
import json
import os
import os.path


# %% functions

def make_meta_file(corpus, corpus_type):
    
    params_list = glob.glob('STM_output/Survey/'+corpus_type+'_params_'+corpus+'/*')
    df_list = []

    # load the data from the mat file
    for params_file in params_list:
        # print(params_file)
        data_dict = sio.loadmat(params_file)
        structure_dict = {field: data_dict['Params'][field][0] for field in data_dict['Params'].dtype.names}
        df = pd.DataFrame(structure_dict)
        df.drop(columns=['x_axis','y_axis'], inplace=True)
        df = df.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
        df = df.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
        df['corpus'] = corpus
        df['corpus_type'] = corpus_type
        df['mat_filename'] = params_file.replace('/Survey/','/MATs/').replace('_params_','_mat_wl4_').replace('_Params.mat', '_MS2024.mat')
        df_list.append(df)
    
    df_all = pd.concat(df_list, ignore_index=True)

    # add the speaker ID and gender info
    if 'MozillaCommonVoice' in corpus:
        valid_df = pd.read_csv('data/speechCorp/'+corpus+'/validated.tsv', sep='\t', quoting=csv.QUOTE_NONE)
        valid_df['path'] = valid_df['path'].str.replace('.mp3', '')
        valid_df.rename(columns={'client_id':'speaker/artist'}, inplace=True)
        df_all = df_all.merge(valid_df[['speaker/artist', 'path', 'gender']], how='left', left_on='filename', right_on='path').drop(columns=['path'])
    elif 'BibleTTS' in corpus:
        df_all['speaker/artist'] = 'BibleTTS_' + df_all['filename'].str[:3]
        df_all['gender'] = np.nan
    elif 'Buckeye' in corpus:
        df_all['speaker/artist'] = df_all['filename'].str[:3].str.replace('s', 'S')
        df_gender = pd.read_csv('data/speechCorp/Buckeye/Buckeye_speaker_info.csv')
        df_all = df_all.merge(df_gender[['SPEAKER', "SPEAKER'S GENDER"]], how='left', left_on='speaker/artist', right_on='SPEAKER').drop(columns=['SPEAKER'])
        df_all.rename(columns={"SPEAKER'S GENDER":'gender'}, inplace=True)
        df_all['speaker/artist'] = 'Buckeye_'+df_all['speaker/artist']
    elif 'EUROM' in corpus:
        df_all['speaker/artist'] = 'EUROM_' + df_all['LangOrInstru'] +'_'+ df_all['filename'].str[:2]
        df_all['gender'] = np.nan
    elif 'HiltonMoser2022_speech' in corpus:
        df_all = preprocHM2022(df_all)
    elif 'MediaSpeech' in corpus:
        df_all['speaker/artist'] = 'MediaSpeech_' + df_all['LangOrInstru'] +'_'+ df_all['filename']
        df_all['gender'] = np.nan
    elif 'LibriSpeech' in corpus:
        split_names = df_all['filename'].str.split('-') # Split the 'name' column by "-"
        first_parts = split_names.str[0] # Extract the first part of the split result
        df_all['speaker/artist'] = first_parts # Add the extracted part as a new column in the DataFrame
        
        ## load LibriSpeech text file
        reader_ids = []
        genders = []
        subsets = []
        durations = []
        names = []
        # Open the file and read line by line
        with open('data/speechCorp/LibriSpeech/SPEAKERS.TXT', 'r') as file:
            # Skip lines starting with ";" (comments) until reaching the data
            while True:
                line = file.readline()
                if not line.startswith(';'):
                    # Start processing data from this line
                    break
            
            # Read the rest of the lines and parse them
            while line:
                # Split each line by "|" character
                data = line.strip().split('|')
                # Extract relevant information
                reader_ids.append(int(data[0].strip()))
                genders.append(data[1].strip())
                subsets.append(data[2].strip())
                durations.append(float(data[3].strip()))
                names.append(data[4].strip())
                # Read the next line
                line = file.readline()
        
        # Create a DataFrame using the lists
        df_LibriSpeech = pd.DataFrame({'reader_id': reader_ids, 'gender': genders, 'subset': subsets, 'duration': durations, 'name': names})
        df_LibriSpeech['reader_id']= df_LibriSpeech['reader_id'].astype(str)
        df_all = df_all.merge(df_LibriSpeech[['reader_id', 'gender']], how='left', left_on='speaker/artist', right_on='reader_id').drop(columns=['reader_id'])
    elif 'primewords_chinese' in corpus:
        with open('data/speechCorp/primewords_chinese/set1_transcript.json', 'r') as file:
            data = json.load(file)
        primewords_df = pd.DataFrame(data)
        primewords_df['file'] = primewords_df['file'].str.replace('.wav', '')
        df_all = df_all.merge(primewords_df[['file', 'user_id']], how='left', left_on='filename', right_on='file').drop(columns=['file'])
        df_all.rename(columns={"user_id":'speaker/artist'}, inplace=True)
        df_all['speaker/artist'] = 'primewords_'+df_all['speaker/artist']
        df_all['gender']=np.nan
    elif 'room_reader' in corpus:
        split_names = df_all['filename'].str.split('_') # Split the 'name' column by "-"
        df_all['speaker/artist'] = split_names.str[1] # Extract the first part of the split result
        RR_df = pd.read_excel('data/speechCorp/room_reader/RoomReader_SessionsEvents.xlsx')
        df_all = df_all.merge(RR_df[['part_ID', 'gender']], how='left', left_on='speaker/artist', right_on='part_ID').drop(columns=['part_ID'])
        df_all['speaker/artist'] = 'RoomReader_'+df_all['speaker/artist']
    elif 'SpeechClarity' in corpus:
        df_all['speaker/artist'] = 'SpeechClarity_'+df_all['filename'].str[:3]
        df_all['gender'] = np.nan
    elif 'TAT-Vol2' in corpus:
        df_all['speaker/artist'] = 'TAT-Vol2_'+df_all['filename'].str[:10]
        df_all['gender'] = df_all['filename'].str[5]
    elif 'thchs30' in corpus:
        split_names = df_all['filename'].str.split('_') # Split the 'name' column by "-"
        df_all['speaker/artist'] = 'thchs30_'+split_names.str[0] # Extract the first part of the split result
        df_all['gender'] = np.nan
    elif 'TIMIT' in corpus:
        split_names = df_all['filepath'].str.split('/') # Split the 'name' column by "-"
        df_all['speaker/artist'] = 'TIMIT_'+split_names.str[-2] # Extract the first part of the split result
        df_all['gender'] = df_all['speaker/artist'].str[6]
    elif 'TTS_Javanese' in corpus:
        df_all['speaker/artist'] = df_all['filename'].str[:9] 
        df_all['gender'] = df_all['filename'].str[2]
    elif 'zeroth' in corpus:
        df_all['speaker/artist'] = df_all['filename'].str[:3] 
        zeroth_df = pd.read_csv('data/speechCorp/zeroth_korean/AUDIO_INFO', sep="|")
        zeroth_df['SPEAKERID'] = zeroth_df['SPEAKERID'].astype(str)
        df_all = df_all.merge(zeroth_df[['SPEAKERID', 'SEX']], how='left', left_on='speaker/artist', right_on='SPEAKERID').drop(columns=['SPEAKERID'])
        df_all.rename(columns={"SEX":'gender'}, inplace=True)
        df_all['speaker/artist'] = 'zeroth_'+df_all['speaker/artist']

    # Music corpora
    elif 'IRMAS' in corpus:
        for nRow in range(len(df_all)):
            txt_path = df_all['filepath'][nRow].replace('.wav', '.txt')
            if os.path.exists(txt_path):
                # Open the file in read mode
                with open(txt_path, 'r') as file:
                    # Read all lines into a list
                    lines = file.readlines()
                converted_lines = [line.strip() for line in lines]
                df_all.loc[nRow,'LangOrInstru'] = '-'.join(converted_lines)
                df_all.loc[nRow,'VoiOrNot'] = int('voi' in converted_lines)
                
        # df_all['speaker/artist'] = '-'.join(df_all['filename'].split('-')[:-1]).strip()
        df_all['speaker/artist'] = df_all['filename'].apply(lambda x: '-'.join(x.split('-')[:-1]).strip())
        df_all['gender'] = np.nan
        df_all['genre'] = np.nan
        
    elif 'Albouy2020Science' in corpus:
        df_all['speaker/artist'] = 'Albouy2020Science'
        df_all['gender'] = 'female'
        df_all['genre'] = 'classical'
        df_all['LangOrInstru'] = 'English'
        df_all.loc[df_all['filename'].str.contains('French', case=False), 'LangOrInstru'] = 'French'
        df_all['VoiOrNot'] = 1

    elif 'CD' in corpus:
        from fuzzywuzzy import process
        df_all['gender'] = np.nan
        def extract_artist(file_path):
            parts = file_path.split('/')
            artist_album_part = parts[3]
            return artist_album_part.split('_')[0]
        
        df_all['speaker/artist'] = df_all['filepath'].apply(extract_artist)
        df_CD = pd.read_excel('data/musicCorp/CD/CD_music_list.xlsx')

        # Function to find the best match for each name in df1 from df2
        def find_best_match(name, choices):
            return process.extractOne(name, choices)
        
        df_all['Best_Match'] = df_all['filename'].apply(lambda x: find_best_match(x, df_CD['Piece'])) # Apply the function to find the best match for each name in df1
        df_all['Matched_Name'] = df_all['Best_Match'].apply(lambda x: x[0]) # Extract matched names and similarity scores
        df_all['Similarity_Score'] = df_all['Best_Match'].apply(lambda x: x[1])
        
        # Join based on matched names
        df_all = pd.merge(df_all, df_CD[['Piece', 'Genre','Instrument']], left_on='Matched_Name', right_on='Piece', how='left')
        df_all['LangOrInstru'] = df_all['Instrument']
        df_all.drop(columns=['Best_Match','Instrument','Matched_Name','Similarity_Score','Piece'], inplace = True)
        df.rename(columns={'Genre': 'genre'}, inplace = True)

        df_all['VoiOrNot'] = 0
        df_all.loc[df_all['LangOrInstru'].str.contains('Voi', case=False), 'VoiOrNot'] = 1

        df_all = df_all[~df_all['filepath'].str.contains('Compilations', case=False)]
        
    elif 'GarlandEncyclopedia' in corpus:
        df_all['speaker/artist'] = df_all['filename']
        df_all['gender'] = np.nan
        df_all['LangOrInstru'] = np.nan
        df_all['genre'] = 'world'
        Garland_novoice_list = list(pd.read_csv('data/musicCorp/GarlandEncyclopedia/Garland_noVoice.csv', header=None)[0])
        df_all['VoiOrNot'] = 1
        df_all.loc[df_all['filename'].isin(Garland_novoice_list), 'VoiOrNot'] = 0
        
    elif 'fma_large' in corpus:
        def revert_numerical_string(num_str):
            return num_str.lstrip('0')
        df_all['filename'] = df_all['filename'].apply(lambda x: revert_numerical_string(x)).astype(int)
        df_tracks = pd.read_csv('data/musicCorp/fma_large/fma_metadata/tracks.csv', low_memory=True, header=1)
        df_tracks.rename(columns={'Unnamed: 0': 'track_id'}, inplace=True)
        df_tracks['track_id'] = pd.to_numeric(df_tracks['track_id'], errors='coerce')
        df_tracks.drop(index=0, inplace=True)
        import ast
        def convert_to_list(string_value):
            return ast.literal_eval(string_value)
        df_tracks['genres'] = df_tracks['genres'].apply(convert_to_list)
        df_all = pd.merge(df_all, df_tracks[['track_id','name','language_code','genre_top', 'genres']], left_on='filename', right_on='track_id', how='left')
        df_all['gender'] = np.nan
        df_all['LangOrInstru'] = df_all['language_code']
        df_all.drop(columns=['language_code','track_id'], inplace=True)
        df_all.rename(columns={'genre_top': 'genre_1', 'name': 'speaker/artist'}, inplace=True)
        df_genres = pd.read_csv('data/musicCorp/fma_large/fma_metadata/genres.csv')
        # if there are more than 1 genres
        nan_rows = list(df_all[df_all['genre_1'].isnull()].index)
        for n_row in nan_rows:
            genre_ids = df_all.loc[n_row, 'genres']
            top_level_values = []
            for id in genre_ids:
                top_level_values.append(df_genres.loc[df_genres['genre_id'] == id, 'top_level'].values[0])
            top_level_values = list(set(top_level_values)) # remove the repeated values
            if len(top_level_values) in [2,3]:
                df_all.loc[n_row, 'genre_1'] = df_genres.loc[df_genres['genre_id'] == top_level_values[0], 'title'].values[0]
                df_all.loc[n_row, 'genre_2'] = df_genres.loc[df_genres['genre_id'] == top_level_values[1], 'title'].values[0]
                if len(top_level_values) == 3:
                    df_all.loc[n_row, 'genre_3'] = df_genres.loc[df_genres['genre_id'] == top_level_values[2], 'title'].values[0]
                else: 
                    df_all.loc[n_row, 'genre_3'] = np.nan
        df_all.drop(columns=['genres'], inplace=True)  

    elif 'Homburg' in corpus:
        split_names = df_all['filename'].str.split('-') # Split the 'name' column by "-"
        df_all['speaker/artist'] = split_names.str[0]
        df_all['gender'] = np.nan
        df_all['LangOrInstru'] = np.nan
        split_filepath = df_all['filepath'].str.split('/') # Split the 'name' column by "-"
        df_all['genre'] = split_filepath.str[3].str.capitalize()
        df_all['VoiOrNot'] = np.nan

    elif 'ismir04_genre' in corpus:
        df_track1 = pd.read_csv('data/musicCorp/ismir04_genre/metadata/development/tracklist.csv', header=None)
        df_track2 = pd.read_csv('data/musicCorp/ismir04_genre/metadata/evaluation/tracklist.csv', header=None)
        df_track3 = pd.read_csv('data/musicCorp/ismir04_genre/metadata/training/tracklist.csv', header=None)
        df_track = pd.concat([df_track1,df_track2,df_track3],ignore_index=True)
        df_track.columns = ['genre', 'speaker/artist', 'album', 'track_name', 'track_num', 'file_name', 'nan']
        split_names = df_track['file_name'].str.split('/') # Split the 'name' column by "-"
        df_track['file_name'] = split_names.str[-1].str.replace('.mp3', '')
        df_all = df_all.merge(df_track[['genre','speaker/artist','file_name']], left_on='filename', right_on='file_name', how='left').drop(columns='file_name')
        df_all['gender'] = np.nan

    elif 'MTG-Jamendo' in corpus:
        df_all['gender'] = np.nan
        df_all['VoiOrNot'] = np.nan
        import sys
        sys.path.append('data/musicCorp/MTG-Jamendo/mtg-jamendo-dataset-github/scripts')
        import commons
        tracks, tags, extra = commons.read_file('data/musicCorp/MTG-Jamendo/mtg-jamendo-dataset-github/data/raw_30s_cleantags_50artists.tsv')
        def get_track_info(filename, tracks):
            try:
                track_dict = tracks[int(filename[:-4])]
                return 'MTG-Jamendo_'+str(track_dict['artist_id']), '/'.join(track_dict['genre']), '/'.join(track_dict['instrument'])
            except:
                return np.nan, np.nan, np.nan
        
        for n in range(len(df_all)):
            df_all.loc[n,'speaker/artist'], df_all.loc[n,'genre'], df_all.loc[n,'LangOrInstru'] = get_track_info(df_all.loc[n,'filename'], tracks)

        genre_dict = dict(
            classical = ['clasical','contemporary','medieval','orchestral','symphonic'],
            electronic = ['club','dance','deephouse','downtempo','dub','dubstep','edm','electronic','electronica','eurodance','house','idm',
                          'lounge','techno','trance','triphop'],
            world = ['african','celtic','ethno','oriental','reggae','ska','tribal','world','worldfusion'],
            rock = ['alternativerock','bluesrock','classicrock','darkwave','ethnicrock','gothic','grunge','hardrock','heavymetal','instrumentalrock',
                    'metal','newwave','poprock','postrock','progressive','punkrock','rock','rocknroll'],
            jazz = ['acidjazz','bossanova','jazz','jazzfusion','swing'],
            blues = ['blues'],
            hiphop = ['alternative', 'breakbeat','hiphop','rap'],
            pop = ['pop','chanson','chillout','electropop','instrumentalpop','psychedelic','synthpop'],
            instrumental = ['ambient','atmospheric','darkambient','newage'],
            experimental = ['experimental','minimal'],
            folk = ['folk','popfolk','singersongwriter'],
            country = ['country'],
            soul_RnB = ['funk','disco','jazzfunk','rnb','soul'],
            easy_listening = ['lounge']
        )
        
        # Define a function to map genre strings to keys
        def map_genre_to_key(genre):
            for key, values in genre_dict.items():
                if genre in values:
                    return key
            return np.nan
        
        for index, row in df_all.iterrows():
            df_all.loc[index,'genre_1'] = np.nan
            df_all.loc[index,'genre_2'] = np.nan
            df_all.loc[index,'genre_3'] = np.nan
        
            if not pd.isna(row['genre']):
                temp_list = row['genre'].split('/')
                top_genre_list=[]
                for item in temp_list:
                    top_genre_list.append(map_genre_to_key(item))
                
                top_genre_list = list(set(top_genre_list))
                top_genre_list = [x for x in top_genre_list if x is not np.nan] # remove np.nan
        
                # print(top_genre_list)
                if len(top_genre_list) in [1,2,3]:
                    df_all.loc[index,'genre_1'] = top_genre_list[0]
                    if len(top_genre_list)>=2:
                        df_all.loc[index,'genre_2'] = top_genre_list[1]
                        if len(top_genre_list)==3:
                            df_all.loc[index,'genre_3'] = top_genre_list[2]
        df_all.drop(columns=['genre'], inplace=True)

    elif 'HiltonMoser2022_song' in corpus:
        df_all = preprocHM2022(df_all)

    elif 'MagnaTagATune' in corpus:
        df_annot = pd.read_csv('data/musicCorp/MagnaTagATune/annotations_final.csv', sep ='\t')
        df_clip = pd.read_csv('data/musicCorp/MagnaTagATune/clip_info_final.csv', sep ='\t')
        
        df_clip['filename'] = df_clip['mp3_path'].str[2:-4]
        df_clip.rename(columns={"artist": "speaker/artist"}, inplace=True)
        
        genre_dict = dict(
            classical = ['clasical', 'classical', 'classic', 'baroque', 'opera', 'operatic', 'female opera', 'male opera', 'medieval'],
            electronic = ['electric', 'electronic', 'electronica', 'electro', 'techno', 'industrial', 'jungle', 'dance', 'trance'],
            world = ['world', 'foreign', 'middle eastern', 'reggae', 'tribal', 'celtic', 'irish', 'arabic', 'eastern', 'oriental', 'india'],
            rock = ['hard rock', 'soft rock', 'rock', 'punk','heavy metal', 'metal', 'heavy'],
            jazz = ['jazz', 'jazzy'],
            blues = ['blues'],
            hiphop = ['rap', 'hip hop'],
            pop = ['pop'],
            instrumental = ['new age', 'ambient'], # combined according to fma
            experimental = ['drone'],
            folk = ['folk'],
            country = ['country'],
            soul_RnB = ['disco','funky', 'funk']
        )
        
        df_genre = pd.DataFrame()
        df_genre['clip_id'] = df_annot['clip_id']
        for key, value in genre_dict.items():
            temp_ind = (df_annot[value]==1).sum(axis=1)>0
            df_genre[key] = temp_ind
        
        # # for visualizing the correlations among genres
        # df_corr = df_genre.drop(columns='clip_id').corr()
        # df_corr.style.background_gradient(cmap='bwr', vmin=-1, vmax=1)
        
        df_genre['genre_counter'] = df_genre.drop(columns='clip_id').sum(axis=1)
        df_genre['genre_1'] = np.nan
        df_genre['genre_2'] = np.nan
        df_genre['genre_3'] = np.nan
        
        for index, row in df_genre.iterrows():
            # Check if genre_counter is less than 3
            if row['genre_counter'] in [1,2,3]:
                # Iterate through each column except 'clip_id' and 'genre_counter'
                g_counter = 0
                for column in df_genre.drop(columns = ['clip_id', 'genre_counter', 'genre_1', 'genre_2', 'genre_3']):
                    # Check if the cell value is True
                    if row[column]:
                        if g_counter == 0:
                            df_genre.loc[index, 'genre_1'] = column
                            g_counter += 1
                        elif g_counter == 1:
                            df_genre.loc[index, 'genre_2'] = column
                            g_counter += 1
                        elif g_counter == 2:
                            df_genre.loc[index, 'genre_3'] = column
        
        df_info = df_clip.merge(df_genre[['clip_id', 'genre_1', 'genre_2', 'genre_3']])
        df_all = df_all.merge(df_info[['speaker/artist', 'filename', 'genre_1', 'genre_2', 'genre_3']], on='filename', how='left')
        df_all['gender']=np.NaN
    
    elif 'NHS2' in corpus:
        df_NHS2_meta = pd.read_csv('data/musicCorp/NHS2/NHS2-metadata.csv')
        df_languoid = pd.read_csv('data/musicCorp/NHS2/glottolog_languoid/languoid.csv')
        df_NHS2_meta = df_NHS2_meta.merge(df_languoid[['id', 'name']], left_on='glottocode', right_on='id')
        df_all = df_all.merge(df_NHS2_meta[['song','name']], left_on='filename', right_on='song', how='left')
        df_all['LangOrInstru'] = df_all['name']
        df_all.drop(columns=['song','name'], inplace=True)
        df_all['genre']='world'
        df_all['gender']=np.NaN
        df_all['VoiOrNot'] = 1
        df_all['speaker/artist'] = df_all['filename']

    
    elif 'SONYC' in corpus:
        df_annot = pd.read_csv('data/envCorp/SONYC/annotations.csv')
        music_speech_columns = [column_name for column_name in df_annot.columns if '6' in column_name or '7' in column_name]
        df_rate_columns = df_annot[music_speech_columns].replace('notsure', np.nan).replace(['near','far'], 1).astype(float).replace(-1, np.nan)
        df_annot_MusicSpeech = pd.concat([df_annot['audio_filename'],df_rate_columns], axis=1)
        df_annot_MusicSpeech['audio_filename'] = df_annot_MusicSpeech['audio_filename'].str.replace('.wav','')
        mean_by_group = pd.DataFrame(np.sum(df_annot_MusicSpeech.groupby('audio_filename').mean()>=0.5, axis=1)>0, columns=['exclude'])
        files_drop = list(mean_by_group[mean_by_group['exclude'] == True].index)
        df_all = df_all[~df_all['filename'].isin(files_drop)]
        df_all['speaker/artist'] = np.nan
        df_all['gender'] = np.nan
        df_all['genre'] = np.nan
        df_all['VoiOrNot'] = 0


    return df_all

def preprocHM2022(df_all):
    # This process will be done for both music and speech
    genre_code = df_all['filename'].str[-1]
    genre_code.replace({
        'A': 'infant-directed',
        'B': 'infant-directed',
        'C': 'adult-directed',
        'D': 'adult-directed'
    }, inplace=True)
    df_all['genre'] = genre_code
    df_all['speaker/artist'] = df_all['filename'].str.slice(0, 5)
    temp_soc = df_all['filename'].str.slice(0, 3)
    temp_soc.replace({
        'MBE': 'Mbendjele',
        'HAD': 'Hadza',
        'NYA': 'Nyangatom',
        'TOP': 'Toposa',
        'BEJ': 'Mandarin-China',
        'JEN': 'Kannada',
        'MEN': 'Mentawai',
        'KRA': 'Polish',
        'LIM': 'Polish',
        'TUR': 'Finnish/Swedish',
        'USD': 'English-US',
        'TOR': 'English-Canada',
        'VAN': 'Bislama',
        'PNG': 'Enga',
        'WEL': 'English-NZ',
        'ARA': 'English-Creole',
        'TSI': 'Tsimane',
        'SPA': 'Quechua/Achuar',
        'QUE': 'Spanish',
        'ACO': 'Spanish',
        'MES': 'Spanish',
    }, inplace=True)
    df_all['LangOrInstru'] = temp_soc
    df_HMmeta = pd.read_csv('data/speechCorp/HiltonMoser2022_speech/stimuli-rawMetadata.csv', usecols=["What is this subject's unique identifier?", "What is the gender of the singer/speaker?"])
    df_HMmeta.rename(columns={
        "What is this subject's unique identifier?": 'speaker/artist', 
        "What is the gender of the singer/speaker?": 'gender'
    }, inplace=True)
    df_HMmeta.replace({
        'Female': 'F',
        'Male': 'M',
        'Not specified': np.nan
    }, inplace=True)
    df_all = df_all.merge(df_HMmeta, on='speaker/artist', how='left')
    return df_all

# %% corpus lists

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

# %% run speech corpora
for corpus in corpus_speech_list:
    savename = 'metaTables/metaData_'+corpus.replace('/', '-')+'.csv'
    if os.path.isfile(savename):
        print('**skipping: '+corpus)
    else:
        print(corpus)
        df_all = make_meta_file(corpus, corpus_type='speech')
        df_all.to_csv(savename)
        
# %% run music corpora  
for corpus in corpus_music_list:
    savename = 'metaTables/metaData_'+corpus.replace('/', '-')+'.csv'
    if os.path.isfile(savename):
        print('**skipping: '+corpus)
    else:
        print(corpus)
        df_all = make_meta_file(corpus, corpus_type='music')
        df_all.to_csv(savename)
      
# %% run environmental sound corpora 
corpus = 'SONYC'
savename = 'metaTables/metaData_'+corpus.replace('/', '-')+'.csv'
if os.path.isfile(savename):
    print('**skipping: '+corpus)
else:
    df_all = make_meta_file(corpus, corpus_type='env')
    df_all.to_csv(savename)
    

# %% Detect overlapped genres
def genre_overlap(df):
    # Initialize an empty dictionary to store genre combinations
    genre_combinations = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract genre values from the row
        genres = [row['genre_1'], row['genre_2'], row['genre_3']]
        # Remove NaN values
        genres = [genre for genre in genres if pd.notnull(genre)]
        # Sort the genres to make sure the combination is unique
        genres.sort()
        # Convert the sorted list of genres to a tuple to use as a dictionary key
        genres_tuple = tuple(genres)
        # Update the count for this combination in the dictionary
        if genres_tuple in genre_combinations:
            genre_combinations[genres_tuple] += 1
        else:
            genre_combinations[genres_tuple] = 1

    return genre_combinations

def sort_dict_by_value(d):
    # Use sorted() function to sort the dictionary based on its values
    sorted_dict = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return sorted_dict

# Example usage:
# Assuming df is your DataFrame containing 'genre_1', 'genre_2', 'genre_3' columns

# corpus_name = 'fma_large'
# corpus_name = 'ismir04_genre'
# corpus_name = 'MagnaTagATune'
corpus_name = 'MTG-Jamendo'

df = pd.read_csv('metaTables/'+'metaData_'+corpus_name+'.csv')
genre_combinations = genre_overlap(df)
genre_combinations_sorted = sort_dict_by_value(genre_combinations)
genre_combinations_sorted