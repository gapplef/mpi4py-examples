#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import scipy.integrate as inte

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("-"*20)
    print(" Running on {:d} cores".format(comm.size))
    print("-"*20)

n_data = comm.size*4
if comm.rank == 0:
    A = np.arange(n_data, dtype=np.float64) # rank 0 has proper data
else:
    A = np.empty(n_data, dtype=np.float64)  # all other just an empty array

# Scatter data in A into A_scatter arrays
n_scatter = n_data//comm.size
A_scatter = np.empty(n_scatter, dtype=np.float64)
comm.Scatter( [A, MPI.DOUBLE], [A_scatter, MPI.DOUBLE] )

print("After Scatter:")
for r in range(comm.size):
    if comm.rank == r:
        print('[{:d}] {}'.format(comm.rank, A_scatter))
    #comm.Barrier()

# Data process function
result_scatter = A_scatter*2
#result_scatter = np.empty_like(A_scatter)
#for i,a in enumerate(A_scatter):
#    f = lambda x: x**a
#    result_scatter[i], _ = inte.quad(f, 1,2)

#result_scatter = np.empty_like(A_scatter)
#def fun(data):
#    result = np.empty_like(data)
#    for i,a in enumerate(data):
#        f = lambda x: x**a
#        result[i], _ = inte.quad(f, 1,2)
#    return result
#result_scatter = fun(A_scatter)


# Allgather data into 'B' and data process result into 'result'
B = np.empty_like(A)
result = np.empty_like(A)
comm.Allgather( [A_scatter, MPI.DOUBLE], [B, MPI.DOUBLE] )
comm.Allgather( [result_scatter, MPI.DOUBLE], [result, MPI.DOUBLE] )

print("After Allgather:")
for r in range(comm.size):
    if comm.rank == r:
        print('[{:d}] {}'.format(comm.rank, B))
        print('[{:d}] {}'.format(comm.rank, result))
    #comm.Barrier()
comm.Barrier()  # synchronization across all group members

