#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print('-'*20)
    print(' Running on %d cores' % comm.size)
    print('-'*20)

# data to be broadcasted
N = 5
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64) # rank 0 has proper data
else:
    A = np.empty(N, dtype=np.float64)  # all other just an empty array

# Broadcast data from rank 0 to all other numbers
comm.Bcast( [A, MPI.DOUBLE] )

# all numbers should now have the same data
print('[%02d] %s' % (comm.rank, A))
comm.Barrier()  # synchronization across all group members
