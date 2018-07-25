import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv

from utilities import read_audio, create_folder, pad_or_trunc
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x
        
        
def calculate_logmel(audio_path, sample_rate, feature_extractor):
    
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
    
    # Normalize energy
    audio /= np.max(np.abs(audio))
    
    # Extract feature
    feature = feature_extractor.transform(audio)
    
    return feature


def read_development_meta(meta_csv):
    
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)
    
    itemids = df['itemid'].tolist()
    datasetids = df['datasetid'].tolist()
    hasbirds = df['hasbird'].tolist()
    
    return itemids, datasetids, hasbirds
    

def read_test_meta(meta_csv):
    
    itemids = []
    
    with open(meta_csv, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
        for li in lis:
            itemid = li[0]
            itemids.append(itemid)
        
    return itemids


def logmel(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    
    # Paths
    if data_type == 'development':
        meta_csv = os.path.join(workspace, 'validation.csv')
        audios_dir = os.path.join(dataset_dir, 'wav')
        
    elif data_type == 'test':
        meta_csv = os.path.join(dataset_dir, 
                                'dcase2018_task3_bird_examplesubmission.csv')
                                
        audios_dir = os.path.join(dataset_dir, 'test_wav')
                                
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'mini_{}.h5'.format(data_type))
        
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 '{}.h5'.format(data_type))
                                 
    create_folder(os.path.dirname(hdf5_path))
    
    # Load data
    if data_type == 'development':
        [itemids, datasetids, hasbirds] = read_development_meta(meta_csv)
        
    elif data_type == 'test':
        itemids = read_test_meta(meta_csv)
    
    # Only use partial data when set mini_data to True
    if mini_data:
        
        audios_num = 300
        random_state = np.random.RandomState(0)
        item_indexes = np.arange(len(itemids))
        random_state.shuffle(item_indexes)
        item_indexes = item_indexes[0 : audios_num]
        
        itemids = [itemids[idx] for idx in item_indexes]
        
        if data_type == 'development':
            datasetids = [datasetids[idx] for idx in item_indexes]
            hasbirds = [hasbirds[idx] for idx in item_indexes]
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    begin_time = time.time()
    
    # hdf5 to write
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='feature', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
    
    # Calculate features for all audio clips
    n = 0
    
    for itemid in itemids:
        
        print(n, itemid)
        
        audio_path = os.path.join(audios_dir, '{}.wav'.format(itemid))
        
        # Extract feature
        feature = calculate_logmel(audio_path=audio_path, 
                                    sample_rate=sample_rate, 
                                    feature_extractor=feature_extractor)
        
        feature = pad_or_trunc(feature, seq_len)
        
        hf['feature'].resize((n + 1, seq_len, mel_bins))
        hf['feature'][n] = feature
        
        n += 1
    
    # Write out meta to hdf5 file
    hf.create_dataset(name='itemid', 
                      data=[s.encode() for s in itemids], 
                      dtype='S60')
        
    if data_type == 'development':
        
        hf.create_dataset(name='datasetid', 
                          data=[s.encode() for s in datasetids], 
                          dtype='S30')
                          
        hf.create_dataset(name='hasbird', 
                          data=hasbirds, 
                          dtype=np.int32)

    hf.close()
                
    print("Write out to {}".format(hdf5_path))
    print("Time: {} s".format(time.time() - begin_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, required=True, choices=['development', 'test'])
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'logmel':
        logmel(args)
        
    else:
        raise Exception("Incorrect arguments!")