import os
import numpy as np
import argparse
import h5py
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
from utilities import (create_folder, get_filename, create_logging, 
                       calculate_auc, calculate_ap, calculate_error, 
                       calculate_scalar, scale)
from models import move_data_to_gpu, init_layer, init_bn, BaselineCnn
import config


def evaluate(model, gen, data_type, max_iteration, cuda):

    (outputs, targets, itemids) = forward(model=model,
                               gen=gen,
                               data_type=data_type,
                               max_iteration=-1,
                               cuda=cuda, 
                               has_target=True)

    error = calculate_error(targets, outputs)
    auc = calculate_auc(targets, outputs)
    ap = calculate_ap(targets, outputs)

    return error, auc, ap


def forward(model, gen, data_type, max_iteration, cuda, has_target):
    """Forward data to a model. 
    
    model: object. 
    gen: object. 
    return_bottleneck: bool. 
    data_type: 'train' | 'validate'. 
    max_iteration: int. 
    cuda: bool. 
    """

    model.eval()

    outputs = []
    targets = []
    itemids = []
    
    iteration = 0

    # Evaluate on mini-batch
    for data in gen.generate_validate(data_type=data_type, max_iteration=max_iteration):
        
        if has_target:
            (batch_x, batch_y, batch_itemids) = data
            targets.append(batch_y)
            
        else:
            (batch_x, batch_itemids) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        batch_output = model(batch_x, return_bottleneck=False)
            
        outputs.append(batch_output.data.cpu().numpy())
        itemids.append(batch_itemids)

        iteration += 1

    outputs = np.concatenate(outputs, axis=0)
    itemids = np.concatenate(itemids, axis=0)
    
    if has_target:
        targets = np.concatenate(targets, axis=0)
        return outputs, targets, itemids
        
    else:
        return outputs, itemids


def train(args):
    
    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    validate = args.validate
    mini_data = args.mini_data
    cuda = args.cuda
    filename = args.filename
    
    batch_size = 64
    fold_for_validation = 0

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'mini_development.h5')
        
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'development.h5')
                                 
    if validate:
        validation_csv = os.path.join(workspace, 'validation.csv')
        
    else:
        validation_csv = None
        fold_for_validation = None

    models_dir = os.path.join(workspace, 'models', filename, 
                              'validation={}'.format(validate))
                              
    create_folder(models_dir)

    # Model
    model = BaselineCnn()

    if cuda:
        model.cuda()

    gen = DataGenerator(hdf5_path=hdf5_path,
                        batch_size=batch_size,
                        validation_csv=validation_csv, 
                        fold_for_validation=fold_for_validation)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    iteration = 0
    train_bgn_time = time.time()

    # Train on mini-batch
    for (batch_x, batch_y) in gen.generate_train():

        # Evaluate
        if iteration % 500 == 0:

            train_fin_time = time.time()

            (tr_error, tr_auc, tr_ap) = evaluate(model=model,
                                                 gen=gen,
                                                 data_type='train',
                                                 max_iteration=-1,
                                                 cuda=cuda)

            (va_error, va_auc, va_ap) = evaluate(model=model,
                                                 gen=gen,
                                                 data_type='validate',
                                                 max_iteration=-1,
                                                 cuda=cuda)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                "iteration: {}, train time: {:.3f} s, validate time: {:.3f} s".format(
                    iteration, train_time, validate_time))
                    
            logging.info(
                "tr_error: {:.3f}, tr_auc: {:.3f}, tr_ap: {:.3f}".format(
                    tr_error, tr_auc, tr_ap))
                    
            logging.info(
                "va_error: {:.3f}, va_auc: {:.3f}, va_ap: {:.3f}".format(
                    va_error, va_auc, va_ap))
                    
            logging.info("")

            train_bgn_time = time.time()

        # Move data to gpu
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Train
        model.train()
        output = model(batch_x)
        loss = F.binary_cross_entropy(output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info("Model saved to {}".format(save_out_path))


def inference_bottleneck(args):

    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    iteration = args.iteration
    cuda = True

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'dev.h5')
    
    model_path = os.path.join(workspace, 'models', filename,
        'md_{}_iters.tar'.format(iteration))
    
    bottleneck_hdf5_path = os.path.join(
        workspace, 'bottlecks', filename, 'bottleneck.h5')
        
    create_folder(os.path.dirname(bottleneck_hdf5_path))

    # Load data
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['feature'][:]
        y = hf['hasbird'][:]
        itemids = hf['itemid'][:]
        datasetids = hf['datasetid'][:]
        folds = hf['fold'][:]

    (mean, std) = calculate_scalar(x)

    samples_num = len(x)

    # Load model
    model = BaselineCnn()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    output_all = []
    bottleneck_all = []

    # Inference on mini-batch
    for n in range(samples_num):
        
        # Transform data
        slice_x = scale(x[n], mean, std)[np.newaxis, :, :]

        # Move data to gpu
        slice_x = move_data_to_gpu(slice_x, cuda, volatile=False)

        # Inference
        model.eval()
        (output, bottleneck) = model(slice_x, return_bottleneck=True)

        output = output.data.cpu().numpy()
        bottleneck = bottleneck.data.cpu().numpy()

        output_all.append(output)
        bottleneck_all.append(bottleneck)

    output_all = np.concatenate(output_all, axis=0)
    bottleneck_all = np.concatenate(bottleneck_all, axis=0)

    # Write bottleneck to hdf5 file
    with h5py.File(bottleneck_hdf5_path, 'w') as hf:
        hf['bottleneck'] = bottleneck_all
        hf['hasbird'] = y
        hf['itemid'] = itemids
        hf['datasetid'] = datasetids
        hf['fold'] = folds
        
    print("Bottleneck saved to {}".format(bottleneck_hdf5_path))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--data_type', type=str)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_inference_bottleneck = subparsers.add_parser('inference_bottleneck')
    parser_inference_bottleneck.add_argument('--workspace', type=str)
    parser_inference_bottleneck.add_argument('--iteration', type=int)

    args = parser.parse_args()

    args.filename = get_filename(__file__)
    
    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_folder(os.path.dirname(logs_dir))
    logging = create_logging(logs_dir, filemode='w')

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_bottleneck':
        inference_bottleneck(args)
