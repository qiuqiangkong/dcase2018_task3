import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import time
import torch

from data_generator import DataGenerator, TestDataGenerator
from utilities import create_folder, get_filename
from models_pytorch import move_data_to_gpu
from main_pytorch import forward, Model


def inference_development_data_bottleneck_features(args):
    
    # Arugments & parameters
    workspace = args.workspace
    validate = args.validate
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    cuda = args.cuda
    
    batch_size = 64
    filename = 'main_pytorch'

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'development.h5')
    
    if validate:
        model_path = os.path.join(workspace, 'models', filename, 
                                'holdout_fold={}'.format(holdout_fold), 
                                'md_{}_iters.tar'.format(iteration))
                                
        bottleneck_hdf5_path = os.path.join(
            workspace, 'bottlenecks', filename, 
            'dev_holdout_fold={}'.format(holdout_fold), 
            '{}_iters'.format(iteration), 'bottleneck.h5')
                                
    else:
        model_path = os.path.join(workspace, 'models', filename, 'full_train', 
                                  'md_{}_iters.tar'.format(iteration))
                              
        bottleneck_hdf5_path = os.path.join(
            workspace, 'bottlenecks', filename, 'dev_full_train', 
            '{}_iters'.format(iteration), 'bottleneck.h5')
   
    create_folder(os.path.dirname(bottleneck_hdf5_path))

    # Load model
    model = Model()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=dev_hdf5_path,
                                batch_size=batch_size, 
                                validation_csv=None, 
                                holdout_fold=None)

    generate_func = generator.generate_validate(
        data_type='train', shuffle=False, max_iteration=None)
    
    # Write bottleneck features
    write_bottleneck_features_to_hdf5(
        model, generate_func, bottleneck_hdf5_path, cuda, return_target=True)


def inference_testing_data_bottleneck_features(args):
    
    # Arugments & parameters
    workspace = args.workspace
    iteration = args.iteration
    cuda = args.cuda
    
    validate = True
    batch_size = 64
    filename = 'main_pytorch'
    
    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'development.h5')
    
    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test.h5')
                              
    model_path = os.path.join(workspace, 'models', filename, 'full_train', 
                                'md_{}_iters.tar'.format(iteration))   
                                
    bottleneck_hdf5_path = os.path.join(
        workspace, 'bottlenecks', filename, 'test_full_train', 
        '{}_iters'.format(iteration), 'bottleneck.h5')
        
    create_folder(os.path.dirname(bottleneck_hdf5_path))

    # Load model
    model = Model()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()
        
    # Data generator
    generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path,
                                    test_hdf5_path=test_hdf5_path, 
                                    batch_size=batch_size)

    generate_func = generator.generate_test()
    
    # Write bottleneck features
    write_bottleneck_features_to_hdf5(
        model, generate_func, bottleneck_hdf5_path, cuda, return_target=False)
    

def write_bottleneck_features_to_hdf5(model, generate_func, 
    bottleneck_hdf5_path, cuda, return_target):

    # Inference
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda,
                   return_target=return_target, 
                   return_bottleneck=True)
                              
    outputs = dict['output']    # (audios_num, 1)
    bottlenecks = dict['bottleneck']    # (audios_num, feature_maps, time_steps)
    itemids = dict['itemid']    # (audios_num,)
    
    if return_target:
        targets = dict['target']    # (audios_num, 1)

    # Write bottleneck to hdf5 file
    with h5py.File(bottleneck_hdf5_path, 'w') as hf:
        
        hf.create_dataset(name='output', data=outputs[:, 0], dtype=np.float32)
        
        hf.create_dataset(name='bottleneck', data=bottlenecks, dtype=np.float32)
        
        hf.create_dataset(name='itemid', data=[s.encode() for s in itemids], 
                          dtype='S60')
                          
        if return_target:
            hf.create_dataset(name='hasbird', data=targets[:, 0], dtype=np.int32)
        
    print("Bottleneck saved to {}".format(bottleneck_hdf5_path))


if __name__ == '__main__':
    """Extract bottleneck features. 
    
    CUDA_VISIBLE_DEVICES=1 python pytorch/inference_bottleneck_features_pytorch.py development --workspace=$WORKSPACE --validate --holdout_fold=1 --iteration=5000 --cuda
    CUDA_VISIBLE_DEVICES=1 python pytorch/inference_bottleneck_features_pytorch.py test --workspace=$WORKSPACE --iteration=5000 --cuda
    """

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_a = subparsers.add_parser('development')
    parser_a.add_argument('--workspace', type=str, required=True)
    parser_a.add_argument('--validate', action='store_true', default=False)
    parser_a.add_argument('--holdout_fold', type=int, choices=[1, 2, 3])
    parser_a.add_argument('--iteration', type=int, required=True)
    parser_a.add_argument('--cuda', action='store_true', default=False)
    
    parser_b = subparsers.add_parser('test')
    parser_b.add_argument('--workspace', type=str, required=True)
    parser_b.add_argument('--iteration', type=int, required=True)
    parser_b.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'development':
        inference_development_data_bottleneck_features(args)
        
    elif args.mode == 'test':
        inference_testing_data_bottleneck_features(args)

    else:
        raise Exception('Incorrect argument!')