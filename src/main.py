# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.haste import HASTEDataset, class_weight
from mean_teacher.vat_loss import VATLoss, entropy, disable_tracking_bn_stats

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

LOG = logging.getLogger('main')

args = None
global_step = 0


def main(context):
    global global_step
    
    global_step = 0
    best_prec1 = 0
    best_ep = 0
    early_stop = 0

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    LOG.info(parameters_string(model))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        
        # train for one epoch
        exploded = train(train_loader, model, ema_model, optimizer, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))

            p1 = prec1 if args.exclude_unlabeled else ema_prec1

            is_best = p1 > best_prec1
            best_prec1 = max(p1, best_prec1)
            best_ep = (epoch + 1) if is_best else best_ep
            early_stop = 0 if is_best else (early_stop + 1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

        if exploded or (early_stop >= 20 and not args.exclude_unlabeled):
            break
            
    result = {'best_acc': best_prec1, 'best_ep': best_ep, 'stop_acc': ema_prec1, 'stop_ep': epoch+1}
    print(result)
    return result


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    if datadir != 'none':
        dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    
        if args.labels:
            with open(args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
            
        dataset_eval = torchvision.datasets.ImageFolder(evaldir, eval_transformation)
    else:
        dataset = HASTEDataset(is_train=True, include_unlabel=not args.exclude_unlabeled, use_mask=args.brain_consistency>0, transform=train_transformation, num_data=args.num_label)
        unlabeled_idxs = dataset.unlabeled_idxs
        labeled_idxs = dataset.labeled_idxs
        dataset_eval = HASTEDataset(is_train=False, include_unlabel=False, use_mask=False, transform=eval_transformation)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch, log):
    global global_step
    #[2.56, 1.0, 2.84]
    class_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(class_weight)).float(), reduction='sum', ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == 'vat':
        consistency_criterion = VATLoss(eps=5.0)
    else:
        assert False, args.consistency_type

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i, ((input, ema_input, mask_input), target) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        target = target.cuda(non_blocking=True)
        minibatch_size = float(len(target))
        labeled_idx = target.ne(NO_LABEL)
        labeled_minibatch_size = float(labeled_idx.sum().item())
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        
        # ema model forward
        with torch.no_grad():
            ema_logit, ema_feat = ema_model(ema_input)
            if args.brain_consistency:
                #with disable_tracking_bn_stats(ema_model):
                ema_mask_logit, ema_mask_feat = ema_model(mask_input)
                    
        # consistency - vat
        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch, args.consistency)
            meters.update('cons_weight', consistency_weight)
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)
        
        if args.consistency and args.consistency_type.startswith('vat'):
            consistency_loss = consistency_weight * consistency_criterion(ema_logit, model, input) / minibatch_size
            meters.update('cons_loss', consistency_loss.data.item())
        
        # mode forward
        logit, feat = model(input)
        if args.brain_consistency:
            mask_logit, mask_feat = model(mask_input[labeled_idx])

        # classification loss
        class_loss = class_criterion(logit, target) / minibatch_size
        if args.brain_consistency:
            class_loss = class_loss + args.brain_weight * class_criterion(mask_logit, target[labeled_idx]) / minibatch_size
        meters.update('class_loss', class_loss.data.item())

        ema_class_loss = class_criterion(ema_logit, target) / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.data.item())
        
        # consistency - non vat
        if args.consistency and not args.consistency_type.startswith('vat'):
            consistency_loss = consistency_weight * consistency_criterion(logit, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.data.item())

        # entropy minimization
        if args.entmin:
            entmin_weight = get_current_consistency_weight(epoch, args.entmin)
            ent_loss = entmin_weight * entropy(logit) / minibatch_size
            meters.update('ent_loss', ent_loss.data.item())
        else:
            ent_loss = 0
            meters.update('ent_loss', 0)

        # brain consistency
        if args.brain_consistency:
            brain_weight = get_current_consistency_weight(epoch, args.brain_consistency)
            if args.brain_consistency_type == 'kl':
                brain_consistency = brain_weight * losses.softmax_kl_loss(logit, ema_mask_logit) / minibatch_size
            else:
                brain_consistency = brain_weight * F.mse_loss(feat, ema_mask_feat, reduction='mean')
            meters.update('brain_cons', brain_consistency.data.item())
        else:
            brain_consistency = 0
            meters.update('brain_cons', 0)
            
        # total loss
        loss = class_loss + consistency_loss + ent_loss + brain_consistency
        
        if (np.isnan(loss.data.item()) or loss.data.item() > 1e5):
            print('Loss explosion: {}'.format(loss.data.item()))
            return True
            
        meters.update('loss', loss.item())

        prec1 = accuracy(logit, target)
        meters.update('top1', prec1.item(), labeled_minibatch_size)

        ema_prec1 = accuracy(ema_logit, target)
        meters.update('ema_top1', ema_prec1.item(), labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Ep: [{0}][{1}/{2}]  '
                'T {meters[batch_time]:.2f}  '
                'Dat {meters[data_time]:.3f}  '
                'Cls {meters[class_loss]:.4f}  '
                'Con {meters[cons_loss]:.4f}  '
                'Ent {meters[ent_loss]:.4f}  '
                'Brn {meters[brain_cons]:.4f}  '
                'Acc {meters[top1]:.2f}'.format(
                    epoch, i, len(train_loader), meters=meters))
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })

    return False


def validate(eval_loader, model, log, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([class_weight])).float(), reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()
    pred_all = []
    target_all = []
    binary_pred_all = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)
        
        #input_var = torch.autograd.Variable(input, volatile=True)
        #target_var = torch.autograd.Variable(target.cuda(non_blocking=True), volatile=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda(non_blocking=True))
    
            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)
    
            # compute output
            model_out = model(input_var)
            if isinstance(model_out, Variable):
                output1 = model_out
            else:
                assert len(model_out) == 2
                output1, _ = model_out
            #softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
            class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1 = accuracy(output1.data, target_var.data, pred_all, target_all, binary_pred_all)
        meters.update('class_loss', class_loss.data.item(), labeled_minibatch_size)
        meters.update('top1', prec1.item(), labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}'.format(
                    i, len(eval_loader), meters=meters))
                    
    pred_all = np.concatenate(pred_all)
    target_all = np.concatenate(target_all)
    binary_pred_all = np.concatenate(binary_pred_all)
    print(classification_report(target_all, pred_all, labels=[0,1,2], target_names=['out', 'good', 'bad']))
    print(confusion_matrix(target_all, pred_all))
    roc = roc_auc_score(target_all == 2, binary_pred_all)
    print(f'roc = {roc}')
    meters.update('roc', roc)

    LOG.info(' * Prec@1 {top1.avg:.3f}'.format(top1=meters['top1']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    

    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    #filename = 'checkpoint.{}.ckpt'.format(epoch)
    #checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    #torch.save(state, checkpoint_path)
    #LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        #shutil.copyfile(checkpoint_path, best_path)
        torch.save(state, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch, weight):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    #return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    return weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, pred_all=None, target_all=None, binary_pred_all=None):
    """Computes the precision@k for the specified values of k"""
    #maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum().float(), 1e-8)
    '''
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    '''
    output = torch.softmax(output, 1)
    pred = torch.argmax(output, 1)
    acc = pred.eq(target).float().sum(0) * 100.0 / labeled_minibatch_size
    if pred_all is not None:
        pred_all.append(pred.cpu().numpy())
        target_all.append(target.cpu().numpy())
        binary_pred_all.append(output.cpu().numpy()[:, -1])

    return acc


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
