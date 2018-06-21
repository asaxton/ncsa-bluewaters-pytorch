from __future__ import print_function
import argparse
import os
import shutil
import time
import copy
import queue
import numpy as np
import gc
import random
random.seed(1)
#from numpy import array
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
recvdata = np.zeros(1,dtype=np.int)
senddata = np.ones(1,dtype=np.int)
comm.Allreduce(senddata, recvdata)
num_workers = int(recvdata[0])
print('num_workers {}'.format(num_workers))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.multiprocessing as mp


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--model', '-m', default='inception_v3',
                    choices=model_names,
                    help='models: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='in Local Steps (default: 10)')

parser.add_argument('--all-reduce-freq-factor',  default=4., type=float,
                    metavar='N', help=('The number of global steps passed'
                                       ' before an allreduce is called is'
                                       ' this factor times number of workers times'
                                       ' --local-step-count-freq (default: 4)'))

parser.add_argument('--local-step-count-freq', default=10, type=int,
                    metavar='N', help='in steps (Default 10)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--num-epochs', '--sn', default=90, type=int,
                    metavar='NS', help='number of steps each worker should step through.')

parser.add_argument('--num-class', default=1000, type=int,
                    metavar='NS', help='number of classification')

parser.add_argument('--checkpoint', type=str, default='model_state.cpt', help='the file where the model will be saved')

args = parser.parse_args()
dist.init_process_group('mpi')
num_workers = dist.get_world_size()
rank = dist.get_rank()

def log_print(*args,**kwargs):
    __builtins__.print('rank %s > ' % rank, *args,**kwargs)

print = log_print

def cnn_validation():

    print("creating model '{}'".format(args.model))
    if args.model == "inception_v3":
        model = models.__dict__[args.model](num_classes=args.num_class)
        IMAGE_SIZE = 299
    elif args.model == "resnet50":
        model = models.__dict__[args.model]()
        IMAGE_SIZE = 224
    elif args.model == "alexnet":
        model = models.__dict__[args.model]()
        IMAGE_SIZE = 224

    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda()

    model.eval()

    if rank == 0:
        torch.save(model.state_dict(), args.checkpoint)

    criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    #criterion = nn.CrossEntropyLoss().cuda()

    validationdir = os.path.join(args.data)

    print("loading dataset step 1'{}'".format(args.data))
    transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])
    validation_dataset = datasets.ImageFolder(validationdir, transform)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    print("loading dataset step 2'{}'".format(args.data))
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, sampler=validation_sampler)

    print("loading dataset done'{}'".format(args.data))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_time = AverageMeter()
    losses = AverageMeter()
    acc_total = AverageMeter()
    acc_correct = AverageMeter()

    print('Pre-Step 1')

    local_step = 0
    last_local_step = 0
    ALL_REDUCE_STEPS = args.local_step_count_freq
    working_ab = 'b'

    print('Initialize Common Model')

    end = time.time()
    # iterations_per_worker = int(len(validation_loader)/float(num_workers))
    iterations_per_worker = int(len(validation_loader))
    print('Starting main loop, {} iterations/eopoch expected'.format(iterations_per_worker))

    for i, (input, target) in enumerate(validation_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #print('step 1')

        #print('step 2')
        input_var = torch.autograd.Variable(input.cuda())
        #print('step 3')
        target_var = torch.autograd.Variable(target.cuda())
        # compute output
        #print('step 4')
        if args.model == "inception_v3": 
            torch.cuda.synchronize()
            output = model(input_var)
            one_hot_tv = make_one_hot(target_var, args.num_class, label_smoothing=0.0)
            o_cpu = output.cpu()
            loss_z = criterion(output, one_hot_tv)
            loss = loss_z/args.batch_size
        elif args.model == "resnet50" or args.model == "alexnet":
            output = model(input_var)
            loss = criterion(output, target_var)

        #print('step 6')
        losses.update(float(loss.data))

        # compute gradient and do SGD step
        #print('step 7')
        local_step += 1

        # top 1 accuracy

        _, predicted = torch.max(output.data, 1)
        acc_correct.update((target_var.data == predicted).sum())
        acc_total.update(target_var.size(0))

        # measure elapsed time

        batch_time.update(time.time() - end)
        if local_step % args.print_freq == 0:
            eps_val = float(args.batch_size)/batch_time.val
            eps_avg = float(args.batch_size)/batch_time.avg
            print('local_step {0} [{1}/{2}] '
                  'avg Time {batch_time.avg:.3f}  '
                  'avg Data {data_time.avg:.3f}  '
                  'l avg example per sec {eps_avg:.3f} '
                  'Top 1 accuracy {acc} '
                  'Loss Avg ({loss.avg:.3f}) Wall Time {wt}'.format(
                    local_step, i+1, iterations_per_worker,
                    batch_time=batch_time,
                    data_time=data_time, 
                    eps_val=eps_val,
                    eps_avg=eps_avg,
                    loss=losses,
                    acc=acc_correct.sum / acc_total.sum,
                    wt=time.ctime()))

            losses.reset()
            data_time.reset()
            batch_time.reset()
            gc.collect()
            end = time.time()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_one_hot(labels, C, label_smoothing=0.0):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
    N x C, where C is class number. One-hot encoded.
    '''
    y = labels.data.cpu()
    y_onehot = torch.FloatTensor(y.size(0), C)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    y_onehot_smoothed = y_onehot * (1 - label_smoothing) + label_smoothing/float(C)
    y_onehot.cuda()
    target = torch.autograd.Variable(y_onehot_smoothed.cuda())
        
    return target

if __name__ == '__main__':
    cnn_validation()

