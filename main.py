import argparse
from ast import parse
import os
import random
import time
import warnings
import sys
import tqdm
import math
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.nn.functional as F
import models
import matplotlib.pyplot as plt
from data import load_data, get_num_classes
from utils import experiment_names, make_dir, Logger, AverageMeter, accuracy, save_checkpoint, source_import, plot_result
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from losses import LDAMLoss, FocalLoss, MixLoss, mixup
from tqdm import tqdm
from copy import deepcopy

import pdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--config', default='./config/cifar10/cifar10.py', type=str)
parser.add_argument('--imb-type', default="exp", type=str, help='imbalance type: {exp, step}')
parser.add_argument('--imb-factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--exp-str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--root-model', type=str, default='checkpoinprepare_folderst')

# {CE, LDAM, focal, CB, mixup, mixup_LDAM}
parser.add_argument('--loss-type', default="CE", type=str, help='loss type: {CE, LDAM, focal, CB, mixup}')
# for focal loss
parser.add_argument('--focal-gamma', default='1.0', type=float, help='hyperparameter for focal loss (default: 1.0)')
# for class balance loss
parser.add_argument('--cb-beta', default='0.9999', type=float, help='hyperparameter for class balance loss (default: 0.9999)')
# for mixup
parser.add_argument('--mixup-beta', default='1.0', type=float, help='hyperparameter for mixup')
# for mixup_LDAM
parser.add_argument('--mixup-LDAM-s', default=30.0, type=float, help='hyperparameter for mixup_LDAM')
# for LOCEloss
parser.add_argument('--loce_coef', default=0.9, type=float)
# {RW, RS, DRW, CB}
parser.add_argument('--train-rule', default='None', type=str, help='data sampling strategy for train loader: {RW, RS, DRW, CB}')

# dual_sampler/soft_weight/augmentation
parser.add_argument('--sampler-type', default='default', help='{default, balance, reverse}')
parser.add_argument('--dual-sample', default=False, type=bool, help='use dual sample in train dataloader')
parser.add_argument('--dual-sampler-type', default='default', help='{default, balance, reverse, reverse_subset, default_subset, def+bal}')
parser.add_argument('--weight-gamma', default=1, type=float, help='smooth the class weight')
parser.add_argument('--use-soft-weight', action='store_true')
parser.add_argument('--use-dual-soft-weight', action='store_true')
parser.add_argument('--weight-type', default='freq', choices=['freq', 'cb'])
parser.add_argument('--augmentation', default='default', type=str, choices=['default', 'randaugment', 'autoaugment'])
parser.add_argument('--use-experts', action='store_true')
parser.add_argument('--use-experts-verbose', action='store_true')
parser.add_argument('--loss-lambda', default=0.5, type=float, help='hyperparameter for multi-experts')
parser.add_argument('--warmup-epoch', default=5, type=float, help='warmup_epoch')

# dual sampler temperature
parser.add_argument('--temp-type', default='', type=str, help='hyperparameter for class-dependent temperature values (RIDE, CDT)')
parser.add_argument('--temp-base', default=1, type=float, help='base temperature for temperature annealing')
parser.add_argument('--temp-eta', default=7, type=float, help='base temperature for temperature annealing, 5 for RIDE')
parser.add_argument('--temp-epsilon', default=0.05, type=float, help='base temperature for temperature annealing, 0.3 for RIDE and CDT')

best_val_acc1 = 0
def main():
    args = parser.parse_args()
    config = source_import(args.config).config

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config['training_opt']['gpu'] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(config['training_opt']['gpu'], ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    global best_val_acc1
    training_opt = config['training_opt']
    log_root = training_opt['log_root'] + '/' + str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    dataset = training_opt['dataset']
    num_classes = training_opt['num_classes']
    data_root = training_opt['data_root']
    optim_params = config['optimizer']['optim_params']
    scheduler_params = config['optimizer']['scheduler_params']
    start_epoch = training_opt['start_epoch']

    if args.warmup_epoch != 5:
        scheduler_params['warmup_epoch'] = args.warmup_epoch
    file_name, log_root = experiment_names(args=args, log_root=log_root, dataset=dataset, arch=config['networks']['arch'])
    file_root = log_root + '/' + file_name
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print('=> create dataset')
    splits = ['train', 'val']
    data = load_data(data_root=data_root, dataset=dataset, num_workers=training_opt['num_workers'],
                     batch_size=training_opt['batch_size'], ngpus_per_node=ngpus_per_node, args=args)
    data = {splits[x]: data[x] for x in range(len(data))}
    args.num_classes = num_classes

    # create model
    print("=> creating model '{}'".format(config['networks']['arch']))
    model = models.__dict__[config['networks']['arch']](**config['networks']['param'])

    if args.gpu is not None:
        if ngpus_per_node == 1:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
    else:
        pass

    # make results root
    make_dir(log_root)
    make_dir(file_root)
    
    # set optimizer and scheduler
    optim_params['params'] = model.parameters()
    optimizer = optim.SGD([optim_params])
    if config['optimizer']['scheduler'] == 'step':
        scheduler = warmup_step_schedule(optimizer, warmup_epoch=scheduler_params['warmup_epoch'],
                                         steps_per_epoch=math.ceil(len(data['train']) / training_opt['accumulation_steps']),
                                         milestones=scheduler_params['step'],
                                         gamma=scheduler_params['gamma'])
    elif config['optimizer']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=math.ceil(len(data['train']) / training_opt['accumulation_steps'])*training_opt['num_epochs'],
                                                        eta_min=scheduler_params['lr_min'])
    else:
        raise ValueError

    acc_list = []
    val_acc_list = []

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume + '/ckpt.pth.tar'):
            print("==> Resuming from checkpoint..")
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume + '/ckpt.pth.tar', map_location='cuda:0')
            start_epoch = checkpoint['epoch']
            best_val_acc1 = checkpoint['best_val_acc1'].cpu()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.resume, 'log.txt'), title=file_name, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise SyntaxError
    else:
        # initial log file for training/validate
        logger = Logger(os.path.join(file_root, 'log.txt'), title=file_name)
        logger.set_names(['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc.', 'Val GM.', 'Major Acc', 'Middle Acc', 'Minor Acc'])

    cudnn.benchmark = True
    cls_num_list = data['train'].dataset.get_cls_num_list()
    print('cls num list:{}'.format(cls_num_list))
    args.cls_num_list = cls_num_list
    args.subset_thr = data['train'].dataset.subset_thr

    with open(os.path.join(log_root, file_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    with open(os.path.join(log_root, file_name, 'config.txt'), 'w') as f:
        f.write(str(config))
    logger.info("***** Running training *****")
    for epoch in range(start_epoch, training_opt['num_epochs']):
        if args.train_rule == 'None':
            train_sampler = None  
            weights_per_cls = None
        elif args.train_rule == 'RS':
            train_sampler = ImbalancedDatasetSampler(data['train'])
            weights_per_cls = None
        elif args.train_rule == 'RW':
            train_sampler = None
            weights_per_cls = 1/np.array(cls_num_list)
            weights_per_cls = weights_per_cls / np.sum(weights_per_cls) * len(cls_num_list)
            weights_per_cls = torch.FloatTensor(weights_per_cls).cuda(args.gpu)
        elif args.train_rule == 'reweight_sqrt':
            weights_per_cls = 1 / np.sqrt(np.array(cls_num_list))
            weights_per_cls = weights_per_cls / np.sum(weights_per_cls) * len(cls_num_list)
            weights_per_cls = torch.FloatTensor(weights_per_cls).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            weights_per_cls = (1.0 - betas[idx]) / np.array(effective_num)
            weights_per_cls = weights_per_cls / np.sum(weights_per_cls) * len(cls_num_list)
            weights_per_cls = torch.FloatTensor(weights_per_cls).cuda(args.gpu)
        elif args.train_rule == 'CB':
            train_sampler = None
            effective_num = 1.0 - np.power(args.cb_beta, cls_num_list)
            weights_per_cls = (1.0 - args.cb_beta) / np.array(effective_num)
            weights_per_cls = weights_per_cls / np.sum(weights_per_cls) * num_classes
            weights_per_cls = torch.tensor(weights_per_cls).float().to(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=weights_per_cls).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=weights_per_cls).cuda(args.gpu)
        elif args.loss_type == 'focal':
            assert args.focal_gamma
            criterion = FocalLoss(weight=weights_per_cls, gamma=args.focal_gamma).cuda(args.gpu)
        elif args.loss_type == 'mixup':
            criterion = MixLoss(args=args, weight=weights_per_cls).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train_loss, train_acc1 = train(data['train'], model, criterion, optimizer, scheduler, epoch, training_opt, args)

        # evaluate on validation set
        if args.use_experts and args.use_experts_verbose:
            val_loss, val_acc1, val_gm, val_major, val_middle, val_minor = validate(
                data['val'], model, nn.CrossEntropyLoss().cuda(args.gpu),
                epoch, file_root, 'val', training_opt, dataset, args)
        else:
            val_loss, val_acc1, val_gm, val_major, val_middle, val_minor = validate(data['val'],
                                                                                                 model, nn.CrossEntropyLoss().cuda(args.gpu),
                                                                                                 epoch, file_root, 'val',
                                                                                                 training_opt, dataset, args)
        logger.append([train_loss, train_acc1, val_loss, val_acc1, val_gm, val_major, val_middle, val_minor])


        # remember best acc@1 and save checkpoint
        is_best = val_acc1 >= best_val_acc1
        best_val_acc1 = max(val_acc1, best_val_acc1)
        print('Best Prec@1: %.3f\n' % (best_val_acc1))
        save_checkpoint(file_root, {
            'epoch': epoch + 1,
            'arch': config['networks']['arch'],
            'state_dict': model.state_dict(),
            'best_val_acc1': best_val_acc1,
            'scheduler': scheduler.state_dict(),
            'optimizer' : optimizer.state_dict()
        }, is_best)
        is_best = False

    logger.set_names(['Best Acc'])
    logger.append([best_val_acc1])
    logger.close()

def train(train_loader, model, criterion, optimizer, scheduler, epoch, training_opt, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.train()
    end = time.time()

    for i, (input, target, input_b, target_b, index, index_b) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # for conventional branch
        if args.gpu is not None:
            input = input[1].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # for re-balancing branch
        if args.dual_sample:
            if args.gpu is not None:
                input_b = input_b[1].cuda(args.gpu, non_blocking=True)
            target_b = target_b.cuda(args.gpu, non_blocking=True)
        else:
            input_b = deepcopy(input).cuda(args.gpu, non_blocking=True)
            target_b = deepcopy(target_b).cuda(args.gpu, non_blocking=True)

        if args.loss_type == 'mixup':
            input_ori, target_ori = input.detach(), target.detach()
            if args.use_experts:
                input, input_b, target, target_b = mixup(input, target, input_b, target_b, args)
            else:
                input, target = mixup(input, target, input_b, target_b, args)
        
        if args.use_experts:
            output_cb = model(input, classifier_cb=True)
            output_rb = model(input_b, classifier_rb=True)
            loss_cb = criterion(output_cb, target)
            loss_rb = criterion(output_rb, target_b)
            loss =  args.loss_lambda * (loss_cb + loss_rb)
        else:
            output = model(input)
            loss = criterion(output, target)
        loss /= training_opt['accumulation_steps']

        # measure accuracy and record loss for debugging
        if args.loss_type == 'mixup':
            if args.use_experts: 
                acc1, acc5 = accuracy((output_rb.cpu() + output_cb.cpu())/2, target_ori.cpu(), topk=(1, 5))
            else: 
                acc1, acc5 = accuracy(model(input_ori).cpu(), target_ori.cpu(), topk=(1, 5))
        else:
            if args.use_experts:
                acc1, acc5 = accuracy((output_rb.cpu() + output_cb.cpu()) / 2, target.cpu(), topk=(1, 5))
            else:
                acc1, acc5 = accuracy(output.cpu(), target.cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if (i+1) % training_opt['accumulation_steps'] == 0 or (i+1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % training_opt['print_freq'] == 0:
            output = ('Epoch: [{0}/{1}][{2}/{3}], lr: {lr:.6f}  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, training_opt['num_epochs'], i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
    
    return losses.avg, top1.avg

def validate(val_loader, model, criterion, epoch, file_root, flag='val', training_opt=None, dataset=None, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    major =  AverageMeter('Major@1', ':6.2f')
    middle = AverageMeter('Middle@1', ':6.2f')
    minor = AverageMeter('Minor@1', ':6.2f')
    if epoch == training_opt['num_epochs']-1 and args.use_experts:
        count = 0
        acc1_list = [0 for _ in range(11)]
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    if args.use_experts and args.use_experts_verbose:
        all_preds_cb = []
        all_preds_rb = []
        losses_cb = AverageMeter('LossCB', ':.4e')
        top1_cb = AverageMeter('AccCB@1', ':6.2f')
        losses_rb = AverageMeter('LossRB', ':.4e')
        top1_rb = AverageMeter('AccRB@1', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.use_experts:
                output_cb, output_rb = model(input, use_experts=args.use_experts)
                output = (output_cb + output_rb) / 2
            else:
                output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.cpu(), target.cpu(), topk=(1, 5))
            if epoch == training_opt['num_epochs']-1 and args.use_experts:
                count += input.size(0)
                acc1_list = [acc1_list[k] + accuracy((output_cb.cpu() * k/10)+(output_rb.cpu() * (1-k/10)), target.cpu(), topk=(1, 5))[0][0] for k in range(11)]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if args.use_experts and args.use_experts_verbose:
                loss_cb = criterion(output_cb, target)
                loss_rb = criterion(output_rb, target)
                losses_cb.update(loss_cb.item(), input.size(0))
                losses_rb.update(loss_rb.item(), input.size(0))

                # measure accuracy and record loss
                acc1_cb, _ = accuracy(output_cb, target, topk=(1, 5))
                acc1_rb, _ = accuracy(output_rb, target, topk=(1, 5))
                top1_cb.update(acc1_cb[0], input.size(0))
                top1_rb.update(acc1_rb[0], input.size(0))

                _, pred_cb = torch.max(output_cb, 1)
                _, pred_rb = torch.max(output_rb, 1)

                all_preds_cb.extend(pred_cb.cpu().numpy())
                all_preds_rb.extend(pred_rb.cpu().numpy())


            if i % training_opt['print_freq'] == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))

        if type(args.cls_num_list) == type(torch.tensor([])):
            cls_num_list = args.cls_num_list.clone().cpu()
        else:
            cls_num_list = args.cls_num_list
        
        # accuracy for major/middle/minor
        if training_opt['dataset'] == 'cifar10':
            major_idx = np.where(np.array(cls_num_list)>=args.subset_thr[0])
            minor_idx = np.where(np.array(cls_num_list)<args.subset_thr[0])
            middle_idx = np.where(np.array(cls_num_list)<args.subset_thr[0])
        else:
            major_idx = np.where(np.array(cls_num_list)>=args.subset_thr[0])
            minor_idx = np.where(np.array(cls_num_list)<args.subset_thr[1])
            middle_idx = np.setdiff1d(np.setdiff1d(range(len(cls_num_list)), major_idx), minor_idx)
        major_acc = cls_acc[major_idx].mean()
        middle_acc = cls_acc[middle_idx].mean()
        minor_acc = cls_acc[minor_idx].mean()
        major.update(major_acc, input.size(0))
        middle.update(middle_acc, input.size(0))
        minor.update(minor_acc, input.size(0))
        print('H:{:.3f} M:{:.3f} T: {:.3f}'.format(major_acc, middle_acc, minor_acc))
        print(out_cls_acc)
        GM = 1
        for i in range(args.num_classes):
            if cls_acc[i] == 0:
                # To prevent the N/A values, we set the minimum value as 0.001
                GM *= (1 / (100 * args.num_classes)) ** (1 / args.num_classes)
            else:
                GM *= (cls_acc[i]) ** (1 / args.num_classes)

    return losses.avg, top1.avg, GM, major_acc, middle_acc, minor_acc

def adjust_learning_rate(optimizer, epoch, args):
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class warmup_step_schedule(optim.lr_scheduler.LambdaLR):
    """
    Linear warmup and then use step function for learning rate scheduling
    """
    def __init__(self, optimizer, warmup_epoch, steps_per_epoch, milestones, gamma=0.01, last_epoch=-1):
        def lr_lambda(step):
            if len(milestones) == 2:
                if step < warmup_epoch * steps_per_epoch:
                    return float(step) / float(max(1.0, warmup_epoch * steps_per_epoch))
                elif step < milestones[0] * steps_per_epoch:
                    return 1
                elif step < milestones[1] * steps_per_epoch:
                    return gamma
                else:
                    return gamma**2
            elif len(milestones) == 1:
                if step < warmup_epoch * steps_per_epoch:
                    return float(step) / float(max(1.0, warmup_epoch * steps_per_epoch))
                elif step < milestones[0] * steps_per_epoch:
                    return 1
                else:
                    return gamma
        super(warmup_step_schedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


if __name__ == '__main__':
    main()
