import numpy as np
import os
import pandas as pd
import argparse


def create_validation(args):
    """Create a validation.csv from three .csv files. 
    """
    
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    folds_num = 3
    
    rs = np.random.RandomState(0)
    
    filenames = ['BirdVoxDCASE20k_csvpublic.csv', 
                 'ff1010bird_metadata_2018.csv', 
                 'warblrb10k_public_metadata_2018.csv']
    
    # Read dataframe
    dataframes = []
    
    for (n, filename) in enumerate(filenames):
        
        filepath = os.path.join(dataset_dir, filename)
        
        df = pd.read_csv(filepath)
        df = pd.DataFrame(df)
        df['fold'] = n + 1
        
        dataframes.append(df)
        
    dataframes = pd.concat(dataframes)
    
    # Write out to csv
    out_path = os.path.join(workspace, 'validation.csv')
    dataframes.to_csv(out_path)
    print("Write out to {}".format(out_path))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    
    args = parser.parse_args()
    
    create_validation(args)