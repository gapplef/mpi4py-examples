#!/usr/bin/env python

"""

How to run:

   mpirun -np <NUM> ./pseudo-whitening <INPUT-IMAGES.h5> <OUTPUT-IMAGES.h5>

"""

import sys
import tables
import numpy as np
from mpi4py import MPI
from numpy.fft import fft2, ifft2
from bernstein.utils import autotable


comm = MPI.COMM_WORLD

in_fname = sys.argv[-2]
out_fname = sys.argv[-1]

try:
    h5in = tables.open_file(in_fname, 'r')
except:
    print("Error: Could not open file %s" % in_fname)
    exit(1)

#h5out = autotable.AutoTable(out_fname)

#
images = h5in.root.images
image_count, height, width = images.shape
image_count = min(image_count, 200)

if comm.rank == 0:
    print("=============================")
    print(" Running %d parallel MPI processes" % comm.size)
    print(" Reading images from '%s'" % in_fname)
    print(" Processing %d images of size %d x %d" % (image_count, width, height))
    print(" Writing whitened images into '%s'" % out_fname)

# Prepare convolution kernel in frequency space
kernel_ = np.zeros((height, width))

# rank 0 needs buffer space to gather data
if comm.rank == 0:
    gbuf = np.empty( (comm.size, height, width) )
else:
    gbuf = None

# Distribute workload so that each MPI process processes image number i, where
#  i % comm.size == comm.rank.
#
# For example if comm.size == 4:
#   rank 0: 0, 4, 8, ...
#   rank 1: 1, 5, 9, ...
#   rank 2: 2, 6, 10, ...
#   rank 3: 3, 7, 11, ...
#
# Each process reads the image from the HDF file by itself. Sadly, python-tables
# does not support parallel writes from multiple processes into the same HDF
# file. So we have to serialize the write operation: Process 0 gathers all
# whitened images and writes them.

comm.Barrier()                      # Start stopwatch ###
t_start = MPI.Wtime()

for i_base in range(0, image_count, comm.size):
    i = i_base + comm.rank
    #
    if i < image_count:
        img  = images[i]            # load image from HDF file
        img_ = fft2(img)            # 2D FFT
        whi_ = img_ * kernel_       # multiply with kernel in freq.-space
        whi  = np.abs(ifft2(whi_))  # inverse FFT back into image space

    # rank 0 gathers whitened images
    comm.Gather(
        [whi, MPI.DOUBLE],   # send buffer
        [gbuf, MPI.DOUBLE],  # receive buffer
        root=0               # rank 0 is root the root-porcess
    )

    # rank 0 has to write into the HDF file
    if comm.rank == 0:
        # Sequentially append each of the images
        for r in range(comm.size):
            pass
            #h5out.append( {'image': gbuf[r]} )

comm.Barrier()
t_diff = MPI.Wtime() - t_start      # Stop stopwatch ###

h5in.close()
# h5out.close()

if comm.rank == 0:
    print(
        " Whitened %d images in %5.2f seconds: %4.2f images per second" % 
            (image_count, t_diff, image_count/t_diff) 
    )
    print("=============================")
