import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging, 
                       calculate_auc, calculate_accuracy, 
                       write_testing_data_submission_csv)
from models_pytorch import move_data_to_gpu, init_layer, BaselineCnn, Vggish
import config


# Hyper-parameteres
Model = Vggish
batch_size = 64


def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate. 
    
    Args:
      model: object
      generator: object
      data_type: string, 'train' | 'validate'
      max_iteration: int, maximum iteration for validation
      cuda: bool
      
    Returns:
      accuracy: float
      auc: float
    """

    generate_func = generator.generate_validate(
        data_type=data_type, shuffle=True,  max_iteration=max_iteration)

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda,
                   return_target=True, 
                   return_bottleneck=False)
                   
    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)

    # Evaluate
    accuracy = calculate_accuracy(targets, outputs)
    auc = calculate_auc(targets, outputs)

    return accuracy, auc


def forward(model, generate_func, cuda, return_target, return_bottleneck):
    """Forward data to a model. 
    
    Args:
      model: object
      generator_func: function
      return_target: bool
      return_bottleneck: bool
      cuda: bool
    
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target', 'bottleneck'
    """

    model.eval()

    outputs = []
    itemids = []
    
    if return_target:
        targets = []
        
    if return_bottleneck:
        bottlenecks = []
    
    # Evaluate on mini-batch
    for data in generate_func:
        
        if return_target:
            (batch_x, batch_y, batch_itemids) = data
            
            
        else:
            (batch_x, batch_itemids) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        if return_bottleneck:
            (batch_output, batch_bottleneck) = model(
                batch_x, return_bottleneck=True)
            
        else:
            batch_output = model(batch_x, return_bottleneck=False)
            
        outputs.append(batch_output.data.cpu().numpy())
        itemids.append(batch_itemids)

        if return_target:
            targets.append(batch_y)
            
        if return_bottleneck:
            bottlenecks.append(batch_bottleneck.data.cpu().numpy())

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    itemids = np.concatenate(itemids, axis=0)
    dict['itemid'] = itemids
    
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    if return_bottleneck:
        dict['bottleneck'] = bottlenecks
        
    return dict


def train(args):
    
    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    validate = args.validate
    holdout_fold = args.holdout_fold
    mini_data = args.mini_data
    cuda = args.cuda
    filename = args.filename

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'mini_development.h5')
        
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'development.h5')
                                 
    if validate:
        validation_csv = os.path.join(workspace, 'validation.csv')
        
        models_dir = os.path.join(workspace, 'models', filename, 
                                'holdout_fold={}'.format(holdout_fold))
        
    else:
        validation_csv = None
        holdout_fold = None

        models_dir = os.path.join(workspace, 'models', filename, 'full_train')
                              
    create_folder(models_dir)

    # Model
    model = Model()

    if cuda:
        model.cuda()

    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              validation_csv=validation_csv, 
                              holdout_fold=holdout_fold)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    # iteration = 0
    train_bgn_time = time.time()

    # Train on mini-batch
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):

        # Evaluate
        if iteration % 500 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_auc) = evaluate(model=model,
                                        generator=generator,
                                        data_type='train',
                                        max_iteration=-1,
                                        cuda=cuda)

            if validate:
                (va_acc, va_auc) = evaluate(model=model,
                                            generator=generator,
                                            data_type='validate',
                                            max_iteration=-1,
                                            cuda=cuda)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            # Print info
            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))
                    
            logging.info(
                'tr_acc: {:.3f}, tr_auc: {:.3f}, '.format(tr_acc, tr_auc))
            
            if validate:
                logging.info(
                    'va_acc: {:.3f}, va_auc: {:.3f}'.format(va_acc, va_auc))
                    
            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
            
        # Reduce learning rate
        if iteration % 100 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.train()
        output = model(batch_x)
        loss = F.binary_cross_entropy(output, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 10000:
            break


def inference_validation(args):

    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    validate = True

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'development.h5')

    validation_csv = os.path.join(workspace, 'validation.csv')


    model_path = os.path.join(workspace, 'models', filename, 
                              'holdout_fold={}'.format(holdout_fold), 
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = Model()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              validation_csv=validation_csv, 
                              holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(
        data_type='validate', shuffle=False, max_iteration=None)

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True, 
                   return_bottleneck=False)
                   
    outputs = dict['output']
    targets = dict['target']
    itemids = dict['itemid']

    # Evaluate
    va_acc = calculate_accuracy(targets, outputs)
    va_auc = calculate_auc(targets, outputs)

    logging.info('va_acc: {:.3f}, va_auc: {:.3f}'.format(va_acc, va_auc))
            
            
def inference_testing_data(args):
    
    # Arugments & parameters
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    validate = True

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'development.h5')
                                 
    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                  'test.h5')
                                 
    model_path = os.path.join(workspace, 'models', filename, 'full_train', 
                              'md_{}_iters.tar'.format(iteration))
                              
    submission_path = os.path.join(workspace, 'submissions', filename, 
                                   'iteration={}'.format(iteration), 
                                   'submission.csv')
                                   
    create_folder(os.path.dirname(submission_path))
    
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
    
    # Inference
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=False, 
                   return_bottleneck=False)
                   
    outputs = dict['output']
    itemids = dict['itemid']
    
    # Write out submission file                             
    write_testing_data_submission_csv(submission_path, itemids, outputs)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int, choices=[1, 2, 3])
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--holdout_fold', type=int, choices=[1, 2, 3])
    parser_inference_validation.add_argument('--iteration', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_testing_data = subparsers.add_parser('inference_testing_data')
    parser_inference_testing_data.add_argument('--workspace', type=str, required=True)
    parser_inference_testing_data.add_argument('--iteration', type=int, required=True)
    parser_inference_testing_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)
    
    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    logging = create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)
        
    elif args.mode == 'inference_testing_data':
        inference_testing_data(args)

    else:
        raise Exception('In correct argument!')