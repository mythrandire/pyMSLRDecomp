"""
testing_block

Multi-scale Low Rank Image Decomposition in Python

Author: Dwiref Oza

Preliminary testing script to test correct operation of decomposition
pipeline.

"""

import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import cv2
from blockSVT import *
import imagesc as imagesc

N = 16
L = np.log2(N)
FOV = (N, N)

sigma = 0

nIter = 1


max_L = L

block_sizes = 2**np.arange(0, int(max_L+1),2)
ms = block_sizes
ns = ms
block_sizes = [(x, x) for x in block_sizes]


levels = len(block_sizes)

bs = np.prod(np.divide(npm.repmat(FOV, levels, 1), block_sizes), 1).astype(np.int)
lambdas = np.sqrt(ms) + np.sqrt(ns) + np.sqrt(np.log2(np.multiply(bs, np.minimum(ms, ns))))
# ms = ns for square input so don't need np.minimum here strictly speaking

A = np.ones((16, 16))

FOVl = FOV + (levels,)
level_dim = len(FOV)


def dummyblock(X, block_size):
    X_shape = X.shape
    halve = lambda x: x/2
    data = makeblocks(X, block_size)
    outdata = [halve(x) for x in data]
    X = buildback(outdata, block_size, X_shape)

    return X

Z_it = np.zeros(FOVl)

Z_it[:,:,0] = A

for it in range(nIter):
    Z_it[:,:,0] = Z_it[:,:,0] + 0.1
    Z_it[:,:,1] = A + 0.01
    for l in range(levels):
        print("iter: ", it, " level: ", l)
        print(Z_it[:,:,l])
        print("changes to: ")
        Z_it[:,:,l] = dummyblock(Z_it[:,:,l], block_sizes[l])
        print("Z_it[:,:,", l, "]: ")
        print(Z_it[:,:,l])
