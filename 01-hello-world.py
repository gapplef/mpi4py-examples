#!/usr/bin/env python

from mpi4py import MPI

comm = MPI.COMM_WORLD

print('Hello! Im rank %d from %d running in total ' % (comm.rank, comm.size))

comm.Barrier()   # synchronization across all group members
# With synchronization on, output order will always from number 0 to 4
# Comment this line out, you'll find the order of output become random
