# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"  # specify which GPU(s) to be used

seed = 123

import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

import logging
import main
import test
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext
import os
import itertools

LOG = logging.getLogger('runner')

# python -m experiments.haste_exp

def parameters():
    defaults = {
        # Technical details
        'workers': 20,
        'checkpoint_epochs': 1,
        'evaluation_epochs': 1,

        # Data
        'dataset': 'haste',
        'exclude_unlabeled': False,  ####
        'num_label': 0,

        # Data sampling
        'base_batch_size': 128, #96
        'base_labeled_batch_size': 32, #24

        # Architecture
        'arch': 'resnet34', #resnext50
        'ema_decay': .994, #0.999

        # Costs
        'consistency_type': 'vat',
        'consistency': 0.5, 
        'consistency_rampup': 5,
        'logit_distance_cost': 0.01 * -1,  ####
        'weight_decay': 5e-5,
        'entmin': 0.1 * 0.0,
        'slice_reg': 0.0,
        'brain_weight': 0.5,
        'brain_consistency': 1.0,
        'brain_consistency_type': 'mse',

        # Optimization
        'epochs': 30, #60
        'lr_rampdown_epochs': 35, #75
        'lr_rampup': 2,
        #'initial_lr': 0.1/250.0,  #####
        'base_lr': 5e-4*4/3,   #5e-4
        'nesterov': True,
    }

    #for num_label, method in itertools.product([1000, 2000, 4000, 0], ['supervised', 'MT', 'MTbrain']):
    for num_label, method in itertools.product([0], ['MTbrain']):
        if method == 'supervised':
            defaults['exclude_unlabeled'] = True
            defaults['num_label'] = num_label
            defaults['consistency_type'] = 'vat'
            defaults['consistency'] = 0.0
            defaults['entmin'] = 0.0
            defaults['brain_consistency'] = 0.0
        elif method == 'MT':
            defaults['exclude_unlabeled'] = False
            defaults['num_label'] = num_label
            defaults['consistency_type'] = 'kl'
            defaults['consistency'] = 1.0
            defaults['entmin'] = 0.0
            defaults['brain_consistency'] = 0.0
        elif method == 'MTVAT':
            defaults['exclude_unlabeled'] = False
            defaults['num_label'] = num_label
            defaults['consistency_type'] = 'vat'
            defaults['consistency'] = 1.0
            defaults['entmin'] = 0.0
            defaults['brain_consistency'] = 0.0
        elif method == 'MTbrain':
            defaults['exclude_unlabeled'] = False
            defaults['num_label'] = num_label
            defaults['consistency_type'] = 'kl'
            defaults['consistency'] = 1.0
            defaults['entmin'] = 1.0
            defaults['brain_consistency'] = 1.0
        else:
            raise Exception('123')

        yield {
            **defaults,
            'title': 'mean teacher haste',
            'data_seed': 123
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    ngpu = torch.cuda.device_count()
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'none',
    }
    if not kwargs['exclude_unlabeled']:
        adapted_args['labeled_batch_size'] = base_labeled_batch_size * ngpu
    else:
        kwargs['epochs'] *= 10
        kwargs['lr_rampdown_epochs'] *= 10
        kwargs['checkpoint_epochs'] *= 10
        kwargs['evaluation_epochs'] *= 10
    args = parse_dict_args(**adapted_args, **kwargs)
    context = RunContext(__file__, data_seed, args)
    main.args = args
    main.main(context)
    
    test.args = args
    return test.test_fn(context)
    #return main.main(context)


if __name__ == "__main__":

    N_repeat = 1 #5
    acc_all = []
    roc_all = []
    
    for run_params in parameters():
    
        for _ in range(N_repeat):
            acc, roc = run(**run_params)
            acc_all.append(acc)
            roc_all.append(roc)
            
        print('acc mean: %f, std: %f' % (np.mean(acc_all), np.std(acc_all)))
        print('roc mean: %f, std: %f' % (np.mean(roc_all), np.std(roc_all)))
