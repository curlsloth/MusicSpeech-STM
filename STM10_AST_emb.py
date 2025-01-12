#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import pandas as pd
import time
import librosa
import sys
import soundfile as sf
from transformers import AutoProcessor, ASTModel
import torch

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

############################ use conda env MusicSpeech-STM_transformers ############################


# import matplotlib.pyplot as plt
# from IPython.display import Audio
# from scipy.io import wavfile

# modified from here: https://www.tensorflow.org/hub/tutorials/yamnet
# Find the name of the class with the top score when mean-aggregated across frames.

# %% corpus lists

def prep_corp_lists():
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
        # 'CD',
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC', 'MacaulayLibrary'] # exclude the 'SONYC_augmented' as there's no wave file
    
    
    # sort the corpora lists to make sure the order is replicable
    corpus_speech_list.sort()
    corpus_music_list.sort()
    corpus_env_list.sort()
    
    return corpus_speech_list+corpus_music_list+corpus_env_list

# %% functions
# verify and convert a loaded audio is on the proper sample_rate (16K), otherwise it would affect the model's results.
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def run_models(df_meta, start_row): # THIS FUNCTION IS DIFFERENT FROM THE OTHERS!
    st = time.time()

    waveform_list = []
    
    for n_row in range(start_row, min(start_row+1000, len(df_meta))): # run 1000 rows at a time, or up to the end of the dataframe
        try:
            filename = df_meta['filepath'].iloc[n_row]
            frame_offset = df_meta['startPoint'].iloc[n_row]-1 # matlab index starts at 1
            frame_end = df_meta['endPoint'].iloc[n_row]
            if corp=='EUROM':
                with open(filename, 'rb') as fid:
                    waveform = np.fromfile(fid, dtype=np.int16)
                waveform = waveform/max(abs(waveform))
                sr = 20000
            elif corp=='MTG-Jamendo': # this corpus is really big, have to use sf to load
                waveform , sr = sf.read(filename, frames=frame_end-frame_offset, start=frame_offset, stop=None, always_2d=True)
                waveform = waveform.mean(axis=1)
            else:
                waveform , sr = librosa.load(filename, sr=None, mono=True)
                
            print("loading success: "+filename)  
            
            if not corp=='MTG-Jamendo': # skip it if is 'MTG-Jamendo'
                waveform = waveform[frame_offset:frame_end]
                
            _, waveform = ensure_sample_rate(sr, waveform) # convert to sr=16000
            
            if (corp=='fma_large') and (n_row in [16606, 58863]): # these 2 files are broken! Use 0 to replace them.
                print("***** using zeros to replace the corrupted audio file: n_row="+str(n_row))
                waveform = np.zeros(16000*4)

        except Exception as e:
            # Print the error message
            print("***** ERROR in n_row="+str(n_row)+ f": {e}")
            waveform = np.zeros(16000*4)
        
        waveform_list.append(waveform)
    
    inputs = processor(waveform_list, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states_batch = outputs.last_hidden_state # [batch_size, num_frames, hidden_size]

    et = time.time()
    print('Execution time:', et - st, 'seconds')
    return last_hidden_states_batch.numpy().mean(axis=1).shape

# %% run AST

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script.py arg1")
        sys.exit(1)

    # Extract command-line arguments
    n = int(sys.argv[1])

    corpus_list = prep_corp_lists()

    corp = corpus_list[n]
    
    metafile = 'metaTables/metaData_'+corp.replace('/', '-')+'.csv'
    df_meta = pd.read_csv(metafile,index_col=0)
    
    embeddings_ast_data_list = []
    
    # as this script is too slow, the data will be run as batches
    for start_row in range(0, len(df_meta), 1000): # every 1000 rows as a batch
        print(start_row) 
        embeddings_ast_data_list.append(run_models(df_meta, start_row))
    
    embeddings_ast_data = np.vstack(embeddings_ast_data_list)
    np.save('ast_output/embeddings/'+corp.replace('/', '-')+'_astEmbeddings.npy', embeddings_ast_data)
