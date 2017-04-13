#!/usr/bin/env python

import numpy as np; np.set_printoptions(linewidth=np.nan)
from mpi4py import MPI as mpi
from parutils import pprint

comm = mpi.COMM_WORLD

pprint('-' * 78)
pprint(' Running on %d cores' % comm.size)
pprint('-' * 78)

comm.Barrier()

# Prepare a vector of N=5 elements to be broadcasted...
N = 5
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)    # rank 0 has proper data
else:
    A = np.empty(N, dtype=np.float64)     # all other just an empty array

# Broadcast A from rank 0 to everybody
comm.Bcast([A, mpi.DOUBLE])

# Everybody should now have the same...
print('[%02d] %s' % (comm.rank, A))
