import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import cv2
from blockSVT import blockSVT
import imagesc as imagesc

N = 64
L = np.log2(N)
FOV = (N, N)

sigma = 0

nIter = 50

rho = 10 # ADMM param

max_L = L

block_sizes = 2**np.arange(0, int(max_L+1),2)
ms = block_sizes
ns = ms
block_sizes = [(x, x) for x in block_sizes]
print(ms)

print("Block sizes: ", block_sizes)

levels = len(block_sizes)

bs = np.prod(np.divide(npm.repmat(FOV, levels, 1), block_sizes), 1).astype(np.int)
print("bs is: ", bs)
lambdas = np.sqrt(ms) + np.sqrt(ns) + np.sqrt(np.log2(np.multiply(bs, np.minimum(ms, ns))))
# ms = ns for square input so don't need np.minimum here strictly speaking


X = cv2.imread('./hanning.png', cv2.IMREAD_GRAYSCALE)
# X = np.ones((16, 16))

FOVl = FOV + (levels,)
level_dim = len(FOV)

A = lambda x : np.sum(x, level_dim) # Summation
temp = np.ones((1, level_dim)).astype(np.int)
temp = (levels,) + [tuple(x) for x in temp.tolist()][0]
AT = lambda x : np.tile(x, temp).T # Adjoint

X_it = np.zeros(FOVl)
Z_it = np.zeros(FOVl)
U_it = np.zeros(FOVl)

for it in range(nIter):
    X_it = 1 / levels * AT(X - A(Z_it - U_it)) + Z_it - U_it
    for l in range(levels):
        Z_it[:,:,l] = blockSVT((X_it[:,:,l] + U_it[:,:,l]), block_sizes[l], (lambdas[l] / rho))


    U_it = U_it - Z_it + X_it

#print(X_it[:,:,0])
plt.imshow(X), plt.show()
imagesc.clean(np.abs(X_it[:,:,0]))
#print(X_it[:,:,0])
imagesc.clean(np.abs(X_it[:,:,1]))
imagesc.clean(np.abs(X_it[:,:,2]))
imagesc.clean(np.abs(X_it[:,:,3]))

print(X_it[:,:,0])
print(X_it[:,:,1])
