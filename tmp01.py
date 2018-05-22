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
                       calculate_scalar, scale, move_data_to_gpu)
from models import init_layer, init_bn, BaselineCnn
import config


def evaluate(model, gen, data_type, max_iteration, cuda):

    (output, target) = forward(model=model,
                               gen=gen,
                               return_bottleneck=False,
                               data_type=data_type,
                               max_iteration=-1,
                               cuda=cuda)

    error = calculate_error(target, output)
    auc = calculate_auc(target, output)
    ap = calculate_ap(target, output)

    return error, auc, ap


def forward(model, gen, return_bottleneck, data_type, max_iteration, cuda):
    """Forward data to a model. 
    
    model: object. 
    gen: object. 
    return_bottleneck: bool. 
    data_type: 'train' | 'validate'. 
    max_iteration: int. 
    cuda: bool. 
    """

    model.eval()

    output_all = []
    target_all = []

    if return_bottleneck:
        bottleneck_all = []

    iteration = 0

    # Evaluate on mini-batch
    for (batch_x, batch_y) in gen.generate_validate(
            data_type=data_type, max_iteration=max_iteration):

        batch_x = move_data_to_gpu(batch_x, cuda, volatile=True)
        batch_y = move_data_to_gpu(batch_y, cuda, volatile=True)

        (batch_output, batch_bottleneck) = model(
            batch_x, return_bottleneck=True)

        output_all.append(batch_output)
        target_all.append(batch_y)

        if return_bottleneck:
            bottleneck_all.append(batch_bottleneck)

        iteration += 1

    output_all = torch.cat(output_all, dim=0)
    target_all = torch.cat(target_all, dim=0)

    output_all = output_all.data.cpu().numpy()
    target_all = target_all.data.cpu().numpy()

    if return_bottleneck:
        bottleneck_all = torch.cat(bottleneck_all, dim=0)
        bottleneck_all = bottleneck_all.data.cpu().numpy()
        return output_all, target_all, bottleneck_all

    else:
        return output_all, target_all


def train(args):

    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    fold_for_validation = config.fold_for_validation

    cuda = True
    scale = True
    batch_size = 64

    # Create log
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode)
    create_folder(os.path.dirname(logs_dir))
    logging = create_logging(logs_dir, filemode='w')

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'dev.h5')
    models_dir = os.path.join(workspace, 'models', filename)
    create_folder(models_dir)

    # Model
    model = BaselineCnn()

    if cuda:
        model.cuda()

    gen = DataGenerator(hdf5_path=hdf5_path,
                        batch_size=batch_size,
                        fold_for_validation=fold_for_validation,
                        scale=scale)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    iteration = 0
    train_bgn_time = time.time()

    # Train on mini-batch
    for (batch_x, batch_y) in gen.generate_train():

        # Evaluate
        if iteration % 100 == 0:

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
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'mini_dev.h5')
    
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

    parser_inference_bottleneck = subparsers.add_parser('inference_bottleneck')
    parser_inference_bottleneck.add_argument('--workspace', type=str)
    parser_inference_bottleneck.add_argument('--iteration', type=int)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_bottleneck':
        inference_bottleneck(args)
