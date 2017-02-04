#!/usr/bin/env python
from mpi4py import MPI as mpi
from __future__ import print_function


comm = mpi.COMM_WORLD

print('Hello! Im rank %d from %d running in total ' % (comm.rank, comm.size))

comm.Barrier()   # wait for everybody to synchronize _here_
