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

parser.add_argument('--print-freq', '-p', default=10, type=int,
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

args = parser.parse_args()
dist.init_process_group('mpi')
num_workers = dist.get_world_size()
rank = dist.get_rank()

ASYNC=False

def log_print(*args,**kwargs):
    __builtins__.print('rank %s > ' % rank, *args,**kwargs)

print = log_print

def async_all_reduce(q_in, q_out, model_grad_copy_ab):
    print('In async_all_reduce. Starting')
    test_tensor = torch.ones([32,3,3])
    test_tensor.cpu()
    while True:
        ab_buff = q_in.get()
        print('ab_buff {}'.format(ab_buff))
        end = time.time()
        if ab_buff is None:
            break
        else:
            model_grad_copy = model_grad_copy_ab[ab_buff]
            dist.all_reduce(test_tensor)
            #allreduce(test_send_tensor, test_recv_tensor)
            print('test_tensor {}'.format(test_tensor))
            #NoneTypeMask = torch.CharTensor([0 if p is None else 1
            #                                 for p in model_grad_copy])
            #dist.all_reduce(NoneTypeMask, op=dist.reduce_op.SUM)
            #for i, (g, ntm) in enumerate(zip(model_grad_copy, NoneTypeMask)):
            #    if ntm == num_workers:
            #        dist.all_reduce(g.data, op=dist.reduce_op.SUM)
            #print('about to do all_reduce loop')
            #for g in model_grad_copy:
            #    print('in all_reduce loop')
            #    if g is not None:
            #        print('doing all_reduce')
            #        dist.all_reduce(g, op=dist.reduce_op.SUM)

            try:
                q_out.get_nowait()
            except queue.Empty:
                pass
            q_out.put(ab_buff)
        print('all_reduce took {}'.format(time.time() - end))
            

def distributed_cnn_train():

    print("creating model '{}'".format(args.model))
    if args.model == "inception_v3": 
        model = models.__dict__[args.model](transform_input=True)
        IMAGE_SIZE = 299
    elif args.model == "resnet50":
        model = models.__dict__[args.model]()
        IMAGE_SIZE = 224
    elif args.model == "alexnet":
        model = models.__dict__[args.model]()
        IMAGE_SIZE = 224

    model.cuda()
    criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    #criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, 
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[2,4,8,16, 32, 64],
                                                     gamma=0.85)
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
        num_workers=0, pin_memory=False, sampler=train_sampler)

    print("loading dataset done'{}'".format(args.data))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    print('switching to training')
    model.train()
    '''
    print('Initialize gradients')
    for input, target in train_loader:
        #print('step 1')
        target = target.cuda(async=True)
        #print('step 2')
        input_var = torch.autograd.Variable(input.cuda())
        #print('step 3')
        target_var = torch.autograd.Variable(target.cuda())
        # compute output
        #print('step 4')
        output, aux_output = model(input_var)

        loss = criterion(output, target_var)
        #print('step 6')
        losses.update(float(loss.data))

        # compute gradient and do SGD step
        #print('step 7')
        optimizer.zero_grad()
        #print('step 8')
        loss.backward()
        #print('step 9')
        break
        
    model_grad_copy_ab = {'a': [None if p.grad is None else torch.ones(p.grad.size()) for p in model.parameters()],
                          'b': [None if p.grad is None else torch.ones(p.grad.size()) for p in model.parameters()]}
    for (a, b), m_p in zip(zip(*model_grad_copy_ab.values()), model.parameters()):
        if a is not None:
            a.cpu()
            #a.share_memory_()
            a.copy_(m_p.grad.data)
    if b is not None:
            b.cpu()
            #b.share_memory_()
            b.copy_(m_p.grad.data)
            '''
    print("setting up async_all_reduce")


    if ASYNC:
        q_in = mp.Queue()
        q_out = mp.Queue()
        async_allreduce_p = mp.Process(target=async_all_reduce, args=(q_in, q_out, model_grad_copy_ab,))
        async_allreduce_p.start()

    print('Pre-Step 1')

    local_step = 0
    last_local_step = 0
    ALL_REDUCE_STEPS = args.local_step_count_freq
    working_ab = 'b'

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
            torch.cuda.synchronize()
            if random.choice([0,1]) == 0:
                for param in model.parameters():
                    dist.all_reduce(param.data, op=dist.reduce_op.MAX)
            else:
                for param in model.parameters():
                    dist.all_reduce(param.data, op=dist.reduce_op.MIN)
            # measure data loading time
            data_time.update(time.time() - end)
            #print('step 1')
            #target = target.cuda(async=True)
            #print('step 2')
            input_var = torch.autograd.Variable(input.cuda())
            #print('step 3')
            target_var = torch.autograd.Variable(target.cuda())
            # compute output
            #print('step 4')
            if args.model == "inception_v3": 
                torch.cuda.synchronize()
                output, aux_output = model(input_var)
                one_hot_tv = make_one_hot(target_var, 1000, label_smoothing=0.5)
                o_cpu = output.cpu()
                loss_z = criterion(output, one_hot_tv)
                loss_aux = criterion(aux_output, one_hot_tv)*0.3
                loss = sum([loss_z, loss_aux])
            elif args.model == "resnet50" or args.model == "alexnet":
                output = model(input_var)
                loss = criterion(output, target_var)

            #print('step 6')
            losses.update(float(loss.data))

            # compute gradient and do SGD step
            #print('step 7')
            optimizer.zero_grad()
            #print('step 8')
            loss.backward()
            #print('step 9')

            #print('step 10')
            #if local_step - last_local_step > ALL_REDUCE_STEPS:
            if ASYNC:
                try:
                    working_ab = q_in.get_nowait()
                except queue.Empty:
                    try:
                        ab_buff = q_out.get_nowait()
                    except queue.Empty:
                        if working_ab == 'a':
                            working_ab = 'b'
                        else:
                            working_ab = 'a'
                        model_grad_copy = model_grad_copy_ab[working_ab]
                        _ = [None if p.grad is None else c.copy_(p.grad.data)
                             for c, p in zip(model_grad_copy, model.parameters())]
                        print('q_in {}'.format(working_ab))
                        q_in.put(working_ab)
                    else:
                        working_ab = ab_buff
                        model_grad_copy = model_grad_copy_ab[ab_buff]
                        torch.cuda.synchronize()
                        for model_p, model_p_copy in zip(model.parameters(), model_grad_copy):
                            if model_p_copy is not None and model_p.grad is not None:
                                model_p.grad.data += model_p_copy
                                model_p.grad.data /= (num_workers + 1)
                            elif model_p_copy is not None or model_p.grad is not None:
                                raise Exception('Model Grads and Sharded grads have inconsistent None values. Go Debug. Aborting')
                            # +1 beacuse there is an extra stale
                            # version of the model in the sum from the all_reduce
                        torch.cuda.synchronize()
                        _ = [None if p.grad is None else c.copy_(p.grad.data)
                             for c, p in zip(model_grad_copy, model.parameters())]
                        print('q_in {}'.format(ab_buff))
                        q_in.put(ab_buff)
                        try:
                            latest_grad_copy = q_in.get_nowait()
                        except queue.Empty:
                            pass
                else:
                    model_grad_copy = model_grad_copy_ab[working_ab]
                    torch.cuda.synchronize()
                    _ = [None if p.grad is None else c.copy_(p.grad.data)
                         for c, p in zip(model_grad_copy, model.parameters())]
                    print('q_in {}'.format(working_ab))
                    q_in.put(working_ab)                
            else:
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

#                model_grad_copy = model_grad_copy_ab['a']
#                _ = [None if p.grad is None else c.copy_(p.grad.data)
#                     for c, p in zip(model_grad_copy, model.parameters())]

#                for grad in model_grad_copy:
#                    if grad is not None:
#                        dist.all_reduce(grad)
#                        #dist.all_reduce(grad.data[0]) # , op=dist.reduce_op.SUM
#                    
#                    _ = [None if c is None else p.grad.data.copy_(c)
#                         for c, p in zip(model_grad_copy, model.parameters())]

            optimizer.step()
            local_step += 1

            # measure elapsed time

            batch_time.update(time.time() - end)
            if local_step % args.print_freq == 0:
                eps_val = float(args.batch_size)/batch_time.val
                eps_avg = float(args.batch_size)/batch_time.avg
                print('local_step {0} [{1}/{2}] '
                      'local_epoch {3} '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                      'l example per sec {eps_val:.3f} ({eps_avg:.3f}) '
                      'Loss {loss.val} ({loss.avg:.3f}) Wall Time {wt}'.format(
                        local_step, i+1, iterations_per_worker, epoch_num,
                        batch_time=batch_time,
                        data_time=data_time, 
                        eps_val=eps_val,
                        eps_avg=eps_avg,
                        loss=losses,
                        wt=time.ctime()))
                      
                losses.reset()
                data_time.reset()
                batch_time.reset()
                gc.collect()
            end = time.time()
            scheduler.step()
            #if i >= iterations_per_worker:
            #    break
    if ASYNC:
        while not q_in.empty():
            q_in.get()
        while not q_out.empty():
            q_out.get()
        q_in.put(None)
        async_allreduce_p.join()

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

