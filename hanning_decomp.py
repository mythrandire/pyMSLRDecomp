"""
hanning_decomp

Multi-scale Low Rank Image Decomposition in Python

Author: Dwiref Oza

Script to run decomposition on an artificial image created using the
Hanning window.

"""


import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import cv2
from blockSVT import blockSVT
from liveplot import liveplot
from liveplot import animate
from matplotlib.colors import Normalize
from skimage.transform import resize


N = 64              # Size of square input
L = np.log2(N)      # No. of scales
FOV = (N, N)        # Shape of input

nIter = 100         # No. of iterations for ADMM

rho = 10            # ADMM param

max_L = L

block_sizes = 2**np.arange(0, int(max_L+1),2)       # block sizes
ms = block_sizes
ns = ms
block_sizes = [(x, x) for x in block_sizes]

print("Block sizes: ", block_sizes)

levels = len(block_sizes)

bs = np.prod(np.divide(npm.repmat(FOV, levels, 1), block_sizes), 1).astype(np.int)
print("bs is: ", bs)
lambdas = np.sqrt(ms) + np.sqrt(ns) + np.sqrt(np.log2(np.multiply(bs, np.minimum(ms, ns))))

# ms = ns for square input so don't need np.minimum here strictly speaking


X = cv2.imread('./fundus.png')[:,:,0]
print(X.shape)
X = resize(X, (64, 64))


print(X)
FOVl = FOV + (levels,)

level_dim = len(FOV)
#plt.imshow(X), plt.show()

A = lambda x : np.sum(x, level_dim) # Summation

AT = lambda x : np.repeat(x[:,:,np.newaxis], levels, axis=2) # Adjoint Operator

X_it = np.zeros(FOVl)
Z_it = np.zeros(FOVl)
U_it = np.zeros(FOVl)


for it in range(nIter):
    X_it = 1 / levels * AT(X - A(Z_it - U_it)) + Z_it - U_it
    liveplot(np.abs(X_it), it)

    for l in range(levels):
        Z_it[:,:,l] = blockSVT((X_it[:,:,l] + U_it[:,:,l]), block_sizes[l], (lambdas[l] / rho))



    U_it = U_it - Z_it + X_it
