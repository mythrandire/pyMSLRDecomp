"""
randshift

Multi-scale Low Rank Image Decomposition in Python

Author: Dwiref Oza

Functions to randomly shift array elements, (record extent of shift)
and to unshift, all using np.roll()

"""


import numpy as np

def randshift(X):

    s = X.shape
    r = np.zeros((len(s))).astype(np.int)

    for  i in range(0, len(s)):
        r[i] = np.random.randint(s[i])

    out = np.roll(X, r)

    return out, r

def randunshift(X, r):
    return np.roll(X, -r)
