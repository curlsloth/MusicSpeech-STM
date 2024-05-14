#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import tensorflow as tf
import scipy
import numpy as np
import pandas as pd
import time
import librosa
import sys
import soundfile as sf

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
        'CD',
        'GarlandEncyclopedia',
        'fma_large',
        'ismir04_genre',
        'MTG-Jamendo',
        'HiltonMoser2022_song',
        'NHS2',
        'MagnaTagATune'
    ]
    
    corpus_env_list = ['SONYC'] # exclude the 'SONYC_augmented' as there's no wave file
    
    
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

def run_models(corp):
    st = time.time()
    YAMNet_path = 'yamnet/1'
    YAMNet_model = tf.saved_model.load(YAMNet_path, tags=None, options=None)
    
    VGGish_path = 'vggish'
    VGGish_model = tf.saved_model.load(VGGish_path, tags=None, options=None)
    
    scores_yamnet_stacked_list = []
    embeddings_yamnet_stacked_list = []
    embeddings_vggish_stacked_list = []
    
    metafile = 'metaTables/metaData_'+corp.replace('/', '-')+'.csv'
    df_meta = pd.read_csv(metafile,index_col=0)
    
    # class_map_path = model.class_map_path().numpy()
    # class_names = class_names_from_csv(class_map_path)
    for n_row in range(len(df_meta)):
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
            
            # YAMNet
            scores_yamnet, embeddings_yamnet, _ = YAMNet_model(waveform) # use YAMNET to score the audio waveform
            scores_yamnet_stacked_list.append(scores_yamnet.numpy().mean(axis=0))
            embeddings_yamnet_stacked_list.append(embeddings_yamnet.numpy().mean(axis=0))
            
            # VGGish
            embeddings_vggish = VGGish_model(waveform)
            embeddings_vggish_stacked_list.append(embeddings_vggish.numpy().mean(axis=0))
            
        except Exception as e:
            # Print the error message
            print("***** ERROR in n_row="+str(n_row)+ f": {e}")
            
    et = time.time()
    print('Execution time:', et - st, 'seconds')
    return np.vstack(scores_yamnet_stacked_list), np.vstack(embeddings_yamnet_stacked_list), np.vstack(embeddings_vggish_stacked_list)

# %% run YAMNet

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script.py arg1")
        sys.exit(1)

    # Extract command-line arguments
    n = int(sys.argv[1])

    corpus_list = prep_corp_lists()

    corp = corpus_list[n]
    
    scores_yamnet_data, embeddings_yamnet_data, embeddings_vggish_data = run_models(corp)
    np.save('yamnet_output/scores/'+corp.replace('/', '-')+'_yamnetScores.npy', scores_yamnet_data)
    np.save('yamnet_output/embeddings/'+corp.replace('/', '-')+'_yamnetEmbeddings.npy', embeddings_yamnet_data)
    np.save('vggish_output/embeddings/'+corp.replace('/', '-')+'_vggishEmbeddings.npy', embeddings_vggish_data)
