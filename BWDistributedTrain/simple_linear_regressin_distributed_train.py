import os
import random
import numpy as np
import socket
import argparse
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.distributed as dist

parser = argparse.ArgumentParser(description='This tool to a sanity check that runs ')
parser.add_argument('--true_slope', default=.3, type=float)
parser.add_argument('--true_intercept', default=-4, type=float)
parser.add_argument('--true_error_delta', default=.5, type=float)

parser.add_argument('--num_train_data', default=10000, type=int)
parser.add_argument('--domain_extents', nargs=2, default=(0,10), type=float)

parser.add_argument('--num_epochs', default=6000, type=int)
parser.add_argument('--num_warmup_epochs', default=100, type=int)
parser.add_argument('--verbose_frequency', default=1000, type=int)
parser.add_argument('--backend', help='mpi, tcp', default='mpi', type=str)


parser.add_argument('--learning_rate', default=.001, type=float)

parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--stop_at_converged_loss', action='store_true')

args = parser.parse_args()



def simple_linear_regression_test():

    TRUE_M = args.true_slope
    TRUE_B = args.true_intercept
    ERROR_DELTA = args.true_error_delta

    NUM_TRAIN_DATA = args.num_train_data
    DOMAIN_EXTENTS = args.domain_extents

    NUM_EPOCHS = args.num_epochs
    NUM_WARMUP_EPOCHS = args.num_warmup_epochs
    VERBOSE_FREQ = args.verbose_frequency
    BACKEND = args.backend

    LEARNING_RATE = args.learning_rate

    USE_GPU = torch.cuda.is_available() and args.use_cuda
    if args.stop_at_converged_loss:
        LOSS_STOP_THRESH = ERROR_DELTA/2. + 0.01
    else:
        LOSS_STOP_THRESH = None

    if os.environ.get('PBS_JOBNAME', False):
        from  mpi4py import MPI
        comm = MPI.COMM_WORLD
        my_ip = socket.gethostbyname(socket.gethostname())
        all_ip = comm.allgather(my_ip)
        size = len(all_ip)
        if  size > 1:
            rank = comm.Get_rank()
            if BACKEND == 'mpi':
                dist.init_process_group('mpi')
            elif BACKEND == 'tcp':
                raise Excpetion('simple_linear_regressin_distributed_train does not support the backend: %s' % BACKEND)
                os.environ['MASTER_ADDR'] = all_ip[0]
                os.environ['MASTER_PORT'] = '29500'
                os.environ['WORLD_SIZE'] = str(size)
                os.environ['RANK'] = str(rank)
                dist.init_process_group('tcp', rank=rank, world_size=size)
            else:
                raise Excpetion('simple_linear_regressin_distributed_train does not support the backend: %s' % BACKEND)
        else:
            rank = 0

    else:
        rank = 0
        size = 1

    x_data = np.array([[random.uniform(*DOMAIN_EXTENTS)] for i in range(NUM_TRAIN_DATA)], dtype=np.float32)
    y_noise = [random.normalvariate(0,ERROR_DELTA) for i in range(NUM_TRAIN_DATA)]
    y_data = np.array([[x*TRUE_M + y_n + TRUE_B] for x, y_n in zip(x_data, y_noise)], dtype=np.float32)

    x_data_local = np.array_split(x_data, size)[rank]
    y_data_local = np.array_split(y_data, size)[rank]

    if USE_GPU:
        print('rank %s: Using CUDA' % rank)
        model = nn.Linear(1,1).cuda()
        x_t_data = torch.autograd.Variable(torch.from_numpy(x_data_local).cuda())
        y_t_data = torch.autograd.Variable(torch.from_numpy(y_data_local).cuda())

    else:
        print('rank %s: Not using CUDA' % rank)
        x_t_data = torch.autograd.Variable(torch.from_numpy(x_data_local))
        y_t_data = torch.autograd.Variable(torch.from_numpy(y_data_local))
        model = nn.Linear(1,1)

    model.weight.data.normal_(0.0, 0.02)
    model.bias.data.fill_(0)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    criterion = nn.MSELoss()

    print('rank %s: Start warmup' % rank)
    for i in range(NUM_EPOCHS + NUM_WARMUP_EPOCHS):
        # print('rank %s: step: %s' % (rank, i))
        if i == NUM_WARMUP_EPOCHS:
            model.weight.data.normal_(0.0, 0.02)
            model.bias.data.fill_(0)
            print('rank %s: End warmup' % rank)
            start_tick = dt.now()
        tick = dt.now()
        optimizer.zero_grad()
        outputs = model(x_t_data)
        loss = criterion(outputs, y_t_data)
        loss.backward()

        optimizer.step()

        if os.environ.get('PBS_JOBNAME', False) and size > 1:
            for param in model.parameters():
                torch.cuda.synchronize()
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                param.grad.data /= float(size)
        tock = dt.now()
        if i % VERBOSE_FREQ == 0:
            print("rank %s: loss: %.6s time_delta: %s" %(rank, float(loss.data), (tock - tick)))
        if LOSS_STOP_THRESH:
            if float(loss.data) < LOSS_STOP_THRESH:
                print("rank %s: loss is less than threshold of %s, stopping" % (rank, LOSS_STOP_THRESH))
                print("rank %s: total steps %s, total time spent: %s" % (rank, i, tock - start_tick))
                break

    print("rank %s: True Bias %.4f, trained Bias %.4f" % (rank, TRUE_B, float(model.bias)))
    print("rank %s: True Slope %.4f, trained Slope %.4f" % (rank, TRUE_M, float(model.weight)))

if __name__ == "__main__":
    simple_linear_regression_test()

        
