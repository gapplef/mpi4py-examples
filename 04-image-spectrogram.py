#!/usr/bin/env python
'''
This example computes the 2D-FFT of every image inside <IMAGES.h5>,
Summs the absolute value of their spectrogram and finally displays
the result in log-scale.
How to run:

   mpirun -n <NUM> python 04-image-spectrogram.py <IMAGES.h5>

'''

import sys
import tables
import pylab
import numpy as np
from numpy.fft import fft2, fftshift
from mpi4py import MPI


comm = MPI.COMM_WORLD

in_fname = sys.argv[-1]

try:
    h5in = tables.open_file(in_fname, 'r')
except:
    print('Error: Could not open file {}'.format(in_fname))
    exit(1)

if comm.rank == 0:
    print(h5in)
    print('==  ' * 5)
    print(h5in.root)

images = h5in.root.images
image_count, height, width = images.shape
image_count = min(image_count, 800)

if comm.rank == 0:
    print('============================')
    print(' Running {:d} parallel MPI processes'.format(comm.size))
    print(' Reading images from {}'.format(in_fname))
    print(' Processing {:d} images of size {:d} x {:d}'.format(image_count, width, height))

# Distribute workload so that each MPI process analyzes image number i, where
#  i % comm.size == comm.rank.
#
# For example if comm.size == 4:
#   rank 0: 0, 4, 8, ...
#   rank 1: 1, 5, 9, ...
#   rank 2: 2, 6, 10, ...
#   rank 3: 3, 7, 11, ...

comm.Barrier()                  # Start stopwatch ###
t_start = MPI.Wtime()

imgs = list(h5in.list_nodes(images))

my_spec = np.zeros((width, height))
if comm.rank == 0:
    print('¤¤  ' * 25)
    print(images._v_attrs)
    print('--  ' * 10)
    print(dir(images))
    print('--  ' * 10)
    # strange this should work images.attrs
    # print(images.attrs)
    print(vars(images))
    print('¤¤  ' * 25)
    print(imgs)


for i in range(comm.rank, image_count, comm.size):
    img = imgs[i]               # Load image from HDF file
    img_ = fft2(img)            # 2D FFT
    my_spec += np.abs(img_)     # Sum absolute value into partial spectrogram

my_spec /= image_count

# Now reduce the partial spectrograms into *spec* by summing
# them all together. The result is only avalable at rank 0.
# If you want the result to be availabe on all processes, use
# Allreduce(...)
spec = np.zeros_like(my_spec)
comm.Reduce(
    [my_spec, MPI.DOUBLE],
    [spec, MPI.DOUBLE],
    op=MPI.SUM,
    root=0
)

comm.Barrier()
t_diff = MPI.Wtime() - t_start # Stop stopwatch ###

h5in.close()

if comm.rank == 0:
    print(' Analyzed {:d} images in {:5.2f} seconds: {:4.2f} images per second'\
           .format(image_count, t_diff, image_count / t_diff))
    print('============================')

# Now rank 0 outputs the resulting spectrogram.
# Either onto the screen or into a image file.
if comm.rank == 0:
    spec = fftshift(spec)
    pylab.imshow(np.log(spec))
    pylab.show()

comm.Barrier()

