import numpy as np
from gen_hanning import *
import matplotlib.pyplot as plt

N = 64              # Size of square input
L = np.log2(N)      # No. of scales
FOV = (N, N)        # Shape of input

max_L = L

block_sizes = 2**np.arange(0, int(max_L+1),2)       # block sizes
block_sizes = [(x, x) for x in block_sizes]

nblocks = np.array([16, 8, 6, 2])

sigma = 10


X, X_decom = gen_hanning(FOV, block_sizes, nblocks, sigma)

print(np.max(X_decom))
print(np.min(X_decom))
plt.imshow(X), plt.show()
