import numpy as np
import os
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time

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
                                        fmin=100., 
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


def logmel(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    
    # Paths
    audio_dir = os.path.join(dataset_dir, 'wav')
    
    validation_csv_path = os.path.join(workspace, 'validation.csv')
    
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'development.h5')
    create_folder(os.path.dirname(hdf5_path))
    
    # Load data
    df = pd.read_csv(validation_csv_path)
    df = pd.DataFrame(df)
    
    audio_num = len(df)
    
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    begin_time = time.time()
    
    # Write out features to hdf5
    with h5py.File(hdf5_path, 'w') as hf:
        
        dt = h5py.special_dtype(vlen=str)
        
        # Reserve space
        hf.create_dataset(name='feature', shape=(audio_num, seq_len, mel_bins), dtype=np.float32)
        hf.create_dataset(name='itemid', shape=(audio_num,), dtype='S50')
        hf.create_dataset(name='datasetid', shape=(audio_num,), dtype='S20')
        hf.create_dataset(name='hasbird', shape=(audio_num,), dtype=np.int32)
        hf.create_dataset(name='fold', shape=(audio_num,), dtype=np.int32)
   
        n = 0
        
        for row in df.iterrows():
    
            itemid = row[1]['itemid']
            datasetid = row[1]['datasetid']
            hasbird = row[1]['hasbird']
            fold = row[1]['fold']
        
            print(n, itemid)
        
            # Calculate feature
            audio_path = os.path.join(audio_dir, '{}.wav'.format(itemid))
            (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
            
            feature = feature_extractor.transform(audio)
            
            feature = pad_or_trunc(feature, seq_len)
    
            hf['feature'][n] = feature
            hf['itemid'][n] = itemid.encode()
            hf['datasetid'][n] = datasetid.encode()
            hf['hasbird'][n] = hasbird
            hf['fold'][n] = fold
            
            if False:
                print(n, itemid, datasetid, hasbird)
                plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
                plt.show()
            
            n += 1
                
    print("Write out to {}".format(hdf5_path))
    print("Time: {} s".format(time.time() - begin_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'logmel':
        logmel(args)
        
    else:
        raise Exception("Incorrect arguments!")