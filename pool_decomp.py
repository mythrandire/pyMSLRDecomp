"""
pool_decomp

Multi-scale Low Rank Image Decomposition in Python

Author: Dwiref Oza

Python Script to compute the multi-scale low rank decomposition of the input
image X. Makes use of multiprocessing.Pool to parallelize the block-wise SVT
computation step of the ADMM solver.

"""


import numpy as np
from multiprocessing import Pool
import numpy.matlib as npm
import matplotlib.pyplot as plt
import cv2
from blockSVT import blockSVT
from liveplot import liveplot
from matplotlib.colors import Normalize
from skimage.transform import resize
from randshift import *


nIter = 100
rho = 10
X = cv2.imread('./fundus.png')[:,:,0]
X = resize(X, (256, 256))
N = 256
L = np.log2(N)
FOV = (N, N)
max_L = L
block_sizes = 2**np.arange(0, int(max_L+1),2)
ms = block_sizes
ns = ms
block_sizes = [(x, x) for x in block_sizes]
print("Block sizes: ", block_sizes)
levels = len(block_sizes)
bs = np.prod(np.divide(npm.repmat(FOV, levels, 1), block_sizes), 1).astype(np.int)
print("bs is: ", bs)
lambdas = np.sqrt(ms) + np.sqrt(ns) + np.sqrt(np.log2(np.multiply(bs, np.minimum(ms, ns))))
FOVl = FOV + (levels,)
level_dim = len(FOV)

A = lambda x : np.sum(x, level_dim) # Summation
AT = lambda x : np.repeat(x[:,:,np.newaxis], levels, axis=2) # Adjoint Operator

X_it = np.zeros(FOVl)
Z_it = np.zeros(FOVl)
U_it = np.zeros(FOVl)

def parSVT(l):
    """
    """
    XU = np.transpose((X_it+U_it), (2, 0, 1))
    XU_s, r = randshift(XU)
    out = blockSVT((XU[l,:,:]), block_sizes[l], (lambdas[l] / rho))
    XU = randunshift(out, r)
    return out

numpools = levels if levels<=10 else 10

for it in range(nIter):
    X_it = 1 / levels * AT(X - A(Z_it - U_it)) + Z_it - U_it

    with Pool(processes=numpools) as pool:
        data = pool.map(parSVT, range(levels))
        pool.close()
        pool.terminate()
        pool.join()
    Z_it = np.reshape(np.asarray(data), (levels, N, N))
    Z_it = np.transpose(Z_it, (1, 2, 0))
    U_it = U_it - Z_it + X_it
    liveplot(np.abs(X_it), it)
