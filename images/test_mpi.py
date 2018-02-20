# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist
import platform


"""
Add /mlodata1/.openmpi/bin to PATH
Add /mlodata1/.openmpi/lib to LD_LIBRARY_PATH

Pytorch need to be built from source. If you use the dmlb environment, it is already done:
    /home/courdier/.conda/envs/dmlb-env/bin/python

Configure passwordless ssh between machine:
    ssh-keygen -b 2048 -t rsa -f $HOME/.ssh/id_rsa -q -N ""
    ssh-copy-id -i ~/.ssh/id_rsa.pub lin@icclusterXXX.iccluster.epfl.ch

For each host, run the following commands.
    echo "StrictHostKeyChecking no" > ~/.ssh/config

mpirun -n 2 --hostfile hostfile --mca btl_tcp_if_exclude docker0,lo --prefix /mlodata1/.openmpi /home/courdier/.conda/envs/dmlb-env/bin/python test_mpi.py
"""


""" All-Reduce example."""


def run_all_reduce(rank, size):
    """ Simple point-to-point communication. """
    group_ids = list(range(size-1))
    group = dist.new_group(group_ids)
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


""" All-Gather example."""


def run_all_gather(rank, size):
    """ Simple point-to-point communication. """
    group_ids = list(range(size-1))
    group = dist.new_group(group_ids)
    tensor = torch.ones(1)
    tensor_list = [torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    dist.all_gather(tensor_list, tensor=tensor, group=group)
    print('Rank ', rank, ' has list ', tensor_list)


""" Broadcast example."""


def run_broadcast(rank, size):
    """ Simple point-to-point communication. """
    group_ids = list(range(size-1))
    group = dist.new_group(group_ids)
    tensor = torch.zeros(1)
    if rank is 0:
        tensor = torch.rand(1)
    dist.broadcast(tensor, src=0, group=group)
    print('Rank ', rank, ' has tensor ', tensor[0])


def run_non_blocking(rank, size):
    tensor = torch.rand(5000, 5000)
    if rank == 0:
        # Send the tensor to process 1
        for dst in range(1, size):
            tensor += 1
            dist.send(tensor=tensor, dst=dst)
            print(
                'Rank ', rank, ' on ', platform.node(),
                'started sending to ', dst)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print('Rank ', rank, ' on ', platform.node(), 'started receiving data')
    print('Rank ', rank, ' has data ', tensor)


def run_blocking(rank, size):
    tensor = torch.zeros(1).cuda()

    if rank == 0:
        # Send the tensor to process 1
        for dst in range(1, size):
            tensor += 1
            dist.send(tensor=tensor, dst=dst)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print('Rank ', rank, ' on ', platform.node(), 'receiving data')
    print('Rank ', rank, ' has data ', tensor[0])


def init_processes(fn):
    """ Initialize the distributed environment. """
    dist.init_process_group('mpi')
    rank = dist.get_rank()
    size = dist.get_world_size()
    torch.cuda.set_device(rank % 2)
    print(
        'I am rank ', rank, ' on ', platform.node(),
        ' with gpu ', torch.cuda.current_device())
    fn(rank, size)


if __name__ == "__main__":
    init_processes(run_blocking)
