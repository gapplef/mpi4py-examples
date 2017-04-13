#!/usr/bin/env python

import numpy as np; np.set_printoptions(linewidth=np.nan)
from mpi4py import MPI as mpi
from parutils import pprint

comm = mpi.COMM_WORLD

pprint('-' * 78)
pprint(' Running on {:d} cores'.format(comm.size))
pprint('-' * 78)

my_N = 4
N = my_N * comm.size

if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)
else:
    A = np.empty(N, dtype=np.float64)

my_A = np.empty(my_N, dtype=np.float64)

# Scatter data into my_A arrays()
comm.Scatter([A, mpi.DOUBLE], [my_A, mpi.DOUBLE])

print('After Scatter:')
for r in range(comm.size):
    if comm.rank == r:
        print('[{:d}] {}'.format(comm.rank, my_A))
    comm.Barrier()

# Everybody is multiplying by 2
my_A *= 2

# Allgather data into A again
comm.Allgather([my_A, mpi.DOUBLE], [A, mpi.DOUBLE])

print('After Allgather:')
for r in range(comm.size):
    if comm.rank == r:
        print('[{:d}] {}'.format(comm.rank, A))
    comm.Barrier()
