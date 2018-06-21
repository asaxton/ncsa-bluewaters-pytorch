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
#import mpi4py
#mpi4py.profile('mpe', logfile='mpi_prof.prof')
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

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='in Local Steps (default: 10)')

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

parser.add_argument('--checkpoint', type=str, default='model_state_dir', help='the directory where the model will be saved')

args = parser.parse_args()
dist.init_process_group('mpi')
num_workers = dist.get_world_size()
rank = dist.get_rank()

def log_print(*args,**kwargs):
    __builtins__.print('rank %s > ' % rank, *args,**kwargs)

print = log_print

def distributed_cnn_train():

    print("creating model '{}'".format(args.model))
    if args.model == "inception_v3":
        model = models.__dict__[args.model](transform_input=True, num_classes=args.num_class)
        IMAGE_SIZE = 299
    elif args.model == "resnet50":
        model = models.__dict__[args.model]()
        IMAGE_SIZE = 224
    elif args.model == "alexnet":
        model = models.__dict__[args.model]()
        IMAGE_SIZE = 224

    model.cuda()

    if rank == 0:
        print('saving model')
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint,'init_model.cpt'))

    criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    #criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, 
                                nesterov=True)

    # decays per iteration
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100,200,400,1000],
                                                     gamma=.85)
    traindir = os.path.join(args.data)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("loading dataset step 1'{}'".format(args.data))
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    print("Distributed Sampler")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    print("loading dataset step 2'{}'".format(args.data))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=False, sampler=train_sampler, drop_last=True)

    print("loading dataset done'{}'".format(args.data))

    load_time = AverageMeter()
    param_pass_time = AverageMeter()
    grad_pass_time = AverageMeter()
    loss_calc_time = AverageMeter()
    backward_time = AverageMeter()
    model_adv_time = AverageMeter()

    losses = AverageMeter()

    # switch to train mode
    print('switching to training')
    model.train()

    print('Pre-Step 1')

    local_step = 0
    last_local_step = 0
    ALL_REDUCE_STEPS = args.local_step_count_freq

    print('Initialize Common Model')
    torch.cuda.synchronize()
    for param in model.parameters():
        if param is not None:
            dist.all_reduce(param.data)
    for param in model.parameters():
        if param is not None:
            param /= float(num_workers)

    end = time.time()
    # iterations_per_worker = int(len(train_loader)/float(num_workers))
    iterations_per_worker = int(len(train_loader))
    print('Starting main loop, {} iterations/eopoch expected'.format(iterations_per_worker))
    for epoch_num in range(args.num_epochs):
        for i, (input, target) in enumerate(train_loader):
            # make model uniform accross all workers
            # common seed accross all woker
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            load_time.update(time.time() - end)

            torch.cuda.synchronize()
            if random.choice([0,1]) == 0:
                for param in model.parameters():
                    dist.all_reduce(param.data, op=dist.reduce_op.MAX)
            else:
                for param in model.parameters():
                    dist.all_reduce(param.data, op=dist.reduce_op.MIN)
            param_pass_time.update(time.time() - end)

            # compute output
            #print('step 4')
            if args.model == "inception_v3": 
                torch.cuda.synchronize()
                output, aux_output = model(input_var)
                one_hot_tv = make_one_hot(target_var, args.num_class, label_smoothing=0.0)
                o_cpu = output.cpu()
                loss_z = criterion(output, one_hot_tv)
                loss_aux = criterion(aux_output, one_hot_tv)*0.3
                loss = sum([loss_z, loss_aux])/(num_workers*args.batch_size)
            elif args.model == "resnet50" or args.model == "alexnet":
                output = model(input_var)
                one_hot_tv = make_one_hot(target_var, args.num_class, label_smoothing=0.5)
                loss = criterion(output, one_hot_tv)/(num_workers*args.batch_size)

            loss_calc_time.update(time.time() - end)
            losses.update(float(loss.data))

            # compute gradient and do SGD step

            optimizer.zero_grad()
            loss.backward()

            backward_time.update(time.time() - end)

            torch.cuda.synchronize()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= float(num_workers)

            for param in model.parameters():
                # common seed accross all woker
                torch.cuda.synchronize()
                if random.choice([0,1]) == 0:
                    dist.all_reduce(param.data, op=dist.reduce_op.MAX)
                else:
                    dist.all_reduce(param.data, op=dist.reduce_op.MIN)

            grad_pass_time.update(time.time() - end)
            optimizer.step()
            local_step += 1
            model_adv_time.update(time.time() - end)
            # measure elapsed time

            if local_step % args.print_freq == 0:
                eps_avg = float(args.batch_size)/model_adv_time.avg
                print('local_step: {loc_step} [{iter}/{iter_p_w}] |'
                      ' epoch: {e_n} |'
                      ' load_time.avg: {load_time.avg:.3f}  |'
                      ' param_pass_time.avg: {param_pass_time.avg:.3f} |'
                      ' loss_calc_time.avg: {loss_calc_time.avg:.3f} |'
                      ' backward_time.avg: {backward_time.avg:.3f} |'
                      ' grad_pass_time.avg: {grad_pass_time.avg:.3f} |'
                      ' model_adv_time.avg: (total_time) {model_adv_time.avg:.3f} |'
                      ' avg example per sec: {eps_avg:.3f} |'
                      ' Loss Avg: {loss.avg:.3f} |'
                      ' Wall Time: {wt}'.format(
                        loc_step=local_step,
                        iter=i+1,
                        iter_p_w=iterations_per_worker,
                        e_n=epoch_num,
                        load_time=load_time,
                        param_pass_time=param_pass_time,
                        grad_pass_time=grad_pass_time,
                        loss_calc_time=loss_calc_time,
                        backward_time=backward_time,
                        model_adv_time=model_adv_time,
                        eps_avg=eps_avg,
                        loss=losses,
                        wt=time.ctime()))
                      
                load_time.reset()
                param_pass_time.reset()
                grad_pass_time.reset()
                loss_calc_time.reset()
                backward_time.reset()
                model_adv_time.reset()
                losses.reset()

                gc.collect()
                if rank == 0:
                    cp_f_d_name = os.path.join(args.checkpoint, 'epoch_{}_iter_{}'.format(epoch_num,
                                                                                           local_step))
                    torch.save(model.state_dict(), cp_f_d_name)

            end = time.time()
            scheduler.step()

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

""" Implementation of a ring-reduce with addition. """
""" http://pytorch.org/tutorials/intermediate/dist_tuto.html """
def allreduce(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = torch.zeros(send.size())
    recv_buff = torch.zeros(send.size())
    accum = torch.zeros(send.size())
    accum[:] = send[:]

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send[:]
        send_req.wait()
    recv[:] = accum[:]

    
def load_data_util(traindir, batch_size):
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torch
    import torch.utils.data
    import torch.utils.data.distributed
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset
#    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

#    train_loader = torch.utils.data.DataLoader(
#        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
#        num_workers=4, pin_memory=True, sampler=train_sampler)

#    return train_dataset, train_sampler, train_loader


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
    distributed_cnn_train()

