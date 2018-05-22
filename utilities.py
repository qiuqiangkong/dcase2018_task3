import numpy as np
import soundfile
import librosa
import os
from sklearn import metrics
import logging

import torch
from torch.autograd import Variable


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name
    
    
def create_logging(log_dir, filemode):
    
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, "%04d.log" % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, "%04d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        raise Exception("Warning: this is a stereo audio!")
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    
    
def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)
        
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    
    return mean, std
   
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
    
def pad_or_trunc(x, max_len):
    print(x.shape)
    if len(x) == max_len:
        return x
    
    elif len(x) > max_len:
        return x[0 : max_len]
        
    else:
        (seq_len, freq_bins) = x.shape
        pad = np.zeros((max_len - seq_len, freq_bins))
        return np.concatenate((x, pad), axis=0)
   
        
def calculate_auc(target, predict):
    return metrics.roc_auc_score(target, predict, average='macro')
    
    
def calculate_ap(target, predict):
    return metrics.average_precision_score(target, predict, average='macro')
    
    
def calculate_error(target, predict):
    binary_predict = (np.sign(predict - 0.5) + 1.) / 2
    error = np.sum(binary_predict != target) / len(target)
    return error
    
    
def move_data_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if cuda:
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x