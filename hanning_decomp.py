import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import cv2
from blockSVT import blockSVT

N = 64
L = np.log2(N)
FOV = (N, N)

sigma = 0

nIter = 50

rho = 10 # ADMM param

max_L = L

block_sizes = 2**np.arange(0,int(max_L+2),2)
ms = block_sizes
ns = ms
block_sizes = [(x, x) for x in block_sizes]

print("Block sizes: ", block_sizes)

levels = len(block_sizes)

bs = np.prod(np.divide(npm.repmat(FOV, levels, 1), block_sizes), 1).astype(np.int)

lambdas = np.sqrt(ms) + np.sqrt(ns) + np.sqrt(np.log2(np.multiply(bs, np.minimum(ms, ns))))


X = cv2.imread('./hanning.png', cv2.IMREAD_GRAYSCALE)
"""
plt.imshow(X)
plt.show()
"""
FOVl = FOV + (levels,)
level_dim = len(FOV)

A = lambda x : np.sum(x, level_dim) # Summation
temp = np.ones((1, level_dim)).astype(np.int)
temp = (levels,) + [tuple(x) for x in temp.tolist()][0]
#print(temp)
#temp2 = (4, 1, 1)
AT = lambda x : np.tile(x, temp).T # Adjoint

X_it = np.zeros(FOVl)
Z_it = np.zeros(FOVl)
U_it = np.zeros(FOVl)

for it in range(nIter):
    #print(X_it.shape)
    new =  X - A(Z_it - U_it)
    #print(new.shape)

    X_it = 1 / levels * AT(X - A(Z_it - U_it)) + Z_it - U_it
    for l in range(levels):
        Z_it[:,:,0] = blockSVT(X_it[:,:,0] + U_it[:,:,0], block_sizes[l], lambdas[l] / rho)

    U_it = U_it - Z_it + X_it
