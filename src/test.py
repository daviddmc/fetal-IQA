import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from mean_teacher import architectures, datasets
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.haste import HASTEDataset
from argparse import Namespace
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

args = None

def create_model(dirpath, use_ema):
    model_factory = architectures.__dict__[args.arch]
    model = model_factory(num_classes=3)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    #if ema:
    for param in model.parameters():
        param.detach_()
    model_path = os.path.join(dirpath, 'best.ckpt')
    sd = torch.load(model_path)['ema_state_dict' if use_ema else 'state_dict']
    model.load_state_dict(sd)
    return model


def accuracy(output, target, input, pred_all=None, target_all=None, binary_pred_all=None, o_pred_all=None):
    """Computes the precision@k for the specified values of k"""
    
    output = torch.softmax(output, dim=1)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum().float(), 1e-8)
    pred = torch.argmax(output, 1)
    acc = pred.eq(target).float().sum(0) * 100.0 / labeled_minibatch_size
    if pred_all is not None:
    
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        output = output.cpu().numpy()
    
        pred_all.append(pred)
        target_all.append(target)
        binary_pred_all.append(output[:, -1])
        o_pred_all.append(output[:, 0])

    return acc

def test_fn(context):

    model = create_model(context.transient_dir, not args.exclude_unlabeled)
    
    dataset_config = datasets.__dict__['haste']()
    
    dataset_eval = HASTEDataset(is_train=False, include_unlabel=False, use_mask=False, transform=dataset_config['eval_transformation'], is_test = True)
    eval_loader = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=20,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False)

    pred_all = []
    target_all = []
    binary_pred_all = []
    o_pred_all = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        
        with torch.no_grad():
            target = target.cuda(non_blocking=True)
    
            minibatch_size = len(target)
    
            # compute output
            model_out = model(input)
            if isinstance(model_out, Variable):
                output1 = model_out
            else:
                assert len(model_out) == 2
                output1, _ = model_out

        # measure accuracy and record loss
        prec1 = accuracy(output1, target, input, pred_all, target_all, binary_pred_all, o_pred_all)
                    
    pred_all = np.concatenate(pred_all)
    target_all = np.concatenate(target_all)
    binary_pred_all = np.concatenate(binary_pred_all)
    o_pred_all = np.concatenate(o_pred_all)
    print(classification_report(target_all, pred_all, labels=[0,1,2], target_names=['out', 'good', 'bad'], digits=5))
    print(confusion_matrix(target_all, pred_all))
    roc = roc_auc_score(target_all == 2, binary_pred_all)
    acc = accuracy_score(target_all, pred_all)
    print(f'roc = {roc}, acc = {acc}')
    return acc, roc

if __name__ == '__main__':

    args = Namespace(
        arch = 'resnet34',
        exclude_unlabeled = False
    )
    context = Namespace(
        transient_dir = './results/haste_exp/2020-01-14_18:15:46/123/transient/'
    )
    test_fn(context)