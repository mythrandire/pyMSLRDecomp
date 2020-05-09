import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import cv2
from blockSVT import blockSVT
import imagesc as imagesc
from liveplot import liveplot
from matplotlib.colors import Normalize
from skimage.transform import resize


N = 64
L = np.log2(N)
FOV = (N, N)

sigma = 0

nIter = 100

rho = 10 # ADMM param

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
# ms = ns for square input so don't need np.minimum here strictly speaking


X = cv2.imread('./fundus.png')[:,:,0]
print(X.shape)
X = resize(X, (64, 64))
# X = np.ones((4, 4))
#X = np.array([[1, 4, 8, 1], [2, 4, 6, 1], [8, 7, 2, 4], [7, 5, 3, 9]])
print(X)
FOVl = FOV + (levels,)

level_dim = len(FOV)
plt.imshow(X), plt.show()

A = lambda x : np.sum(x, level_dim) # Summation

AT = lambda x : np.repeat(x[:,:,np.newaxis], levels, axis=2) # Adjoint Operator

X_it = np.zeros(FOVl)
Z_it = np.zeros(FOVl)
U_it = np.zeros(FOVl)

#X_it = 1 / levels * AT(X - A(Z_it - U_it)) + Z_it - U_it

#print(X_it[:,:,0])
#print(X_it[:,:,1])

for it in range(nIter):
    X_it = 1 / levels * AT(X - A(Z_it - U_it)) + Z_it - U_it
    # the bug is in the line above
    # ergo the discrepenency is in one of the lambda functions: A or AT
    for l in range(levels):
        Z_it[:,:,l] = blockSVT((X_it[:,:,l] + U_it[:,:,l]), block_sizes[l], (lambdas[l] / rho))
        #print('iter: ', it, ' level:', l)
        #print('Z_it[:,:,', l, ']: ', Z_it[:,:,l])


    U_it = U_it - Z_it + X_it


liveplot(np.abs(X_it))
#print(X_it[:,:,0] == X_it[:,:,1])
