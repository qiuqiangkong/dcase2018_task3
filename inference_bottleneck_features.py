import os
import numpy as np
import argparse
import h5py
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
from utilities import create_folder, get_filename
from models import move_data_to_gpu, BaselineCnn


def forward_bottleneck(model, generate_func, cuda):
    """Forward data to a model. 
    
    model: object. 
    generator_func: function. 
    return_bottleneck: bool. 
    cuda: bool. 
    """

    model.eval()

    outputs = []
    bottlenecks = []
    targets = []
    itemids = []
    
    iteration = 0

    # Evaluate on mini-batch
    for data in generate_func:
        
        (batch_x, batch_y, batch_itemids) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        (batch_output, batch_bottleneck) = model(
            batch_x, return_bottleneck=True)

        outputs.append(batch_output.data.cpu().numpy())
        bottlenecks.append(batch_bottleneck.data.cpu().numpy())
        targets.append(batch_y)
        itemids.append(batch_itemids)

        iteration += 1

    outputs = np.concatenate(outputs, axis=0)
    bottlenecks = np.concatenate(bottlenecks, axis=0)
    targets = np.concatenate(targets, axis=0)
    itemids = np.concatenate(itemids, axis=0)
    
    return outputs, bottlenecks, targets, itemids


def inference_bottleneck_features(args):
    
    workspace = args.workspace
    iteration = args.iteration
    cuda = args.cuda
    
    # Arugments & parameters
    workspace = args.workspace
    iteration = args.iteration
    cuda = args.cuda

    validate = True
    batch_size = 64
    filename = 'main_pytorch'

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'development.h5')

    model_path = os.path.join(workspace, 'models', filename, 
                              'validation={}'.format(validate), 
                              'md_{}_iters.tar'.format(iteration))
                              
    bottleneck_hdf5_path = os.path.join(
        workspace, 'bottlecks', filename, 'bottleneck.h5')
        
    create_folder(os.path.dirname(bottleneck_hdf5_path))

    # Load model
    model = BaselineCnn()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size, 
                              validation_csv=None, 
                              fold_for_validation=None)

    generate_func = generator.generate_validate(data_type='train')
    
    # Inference
    (outputs, bottlenecks, targets, itemids) = forward_bottleneck(
        model=model,
        generate_func=generate_func, 
        cuda=cuda)
    '''
    outputs: (N, 1)
    bottlenecks: (N, feature_maps, time_steps) = (N, 128, 15)
    targets: (N, 1)
    itemids: (N,)
    '''

    # Write bottleneck to hdf5 file
    with h5py.File(bottleneck_hdf5_path, 'w') as hf:
        
        hf.create_dataset(name='output', data=outputs[:, 0], dtype=np.float32)
        
        hf.create_dataset(name='bottleneck', data=bottlenecks, dtype=np.float32)
        
        hf.create_dataset(name='hasbird', data=targets[:, 0], dtype=np.int32)
        
        hf.create_dataset(name='itemid', data=[s.encode() for s in itemids], 
                          dtype='S60')
        
    print("Bottleneck saved to {}".format(bottleneck_hdf5_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    
    inference_bottleneck_features(args)

    