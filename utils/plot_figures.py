import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd

from features import LogMelExtractor, calculate_logmel
import config


def plot_logmel(args):
    """Plot log Mel feature of one audio per class. 
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    
    plot_num = 12
    
    # Paths
    meta_csv = os.path.join(workspace, 'validation.csv')
    audios_dir = os.path.join(dataset_dir, 'wav')
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    

    # Calculate log mel feature of audio clips
    df = pd.read_csv(meta_csv, sep=',')
    df = pd.DataFrame(df)
    
    n = 0
    itemids = []
    features = []
    hasbirds = []
    
    for row in df.iterrows():
        
        if n == 12:
            break
        
        itemid = row[1]['itemid']
        hasbird = row[1]['hasbird']
    
        audio_path = os.path.join(audios_dir, '{}.wav'.format(itemid))
        
        feature = calculate_logmel(audio_path=audio_path, 
                                   sample_rate=sample_rate, 
                                   feature_extractor=feature_extractor)
                
        itemids.append(itemid)
        features.append(feature)
        hasbirds.append(hasbird)
        
        n += 1
        
    # Plot
    rows_num = 3
    cols_num = 4
    n = 0
    
    fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))
    
    for n in range(plot_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].matshow(features[n].T, origin='lower', aspect='auto', 
                              cmap='jet', vmin=-10, vmax=-2)
        axs[row, col].set_title('No. {}, hasbird={}'.format(n, hasbirds[n]))
        axs[row, col].set_ylabel('log mel bins')
        axs[row, col].yaxis.set_ticks([])
        axs[row, col].xaxis.set_ticks([0, seq_len])
        axs[row, col].xaxis.set_ticklabels(['0', '10 s'], fontsize='small')
        axs[row, col].xaxis.tick_bottom()
    
    for n in range(plot_num, rows_num * cols_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].set_visible(False)
    
    for (n, itemid) in enumerate(itemids):
        print('No. {}, {}.wav'.format(n, itemid))
    
    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    """Plot logmel. 
    
    Example: python utils/plot_figures.py plot_logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
    """
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot_logmel = subparsers.add_parser('plot_logmel')
    parser_plot_logmel.add_argument('--dataset_dir', type=str)
    parser_plot_logmel.add_argument('--workspace', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'plot_logmel':
        plot_logmel(args)
        
    else:
        raise Exception("Incorrect arguments!")