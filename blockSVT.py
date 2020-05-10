"""
blockSVT

Multi-scale Low Rank Image Decomposition in Python

Author: Dwiref Oza

Functions to simulate behavior of blockproc from MATLAB and to compute
block-wise Singular Value Threshold (SVT)
"""


import numpy as np
from skimage.util.shape import view_as_blocks
from SVT import SVT
from SoftThresh import SoftThresh


def makeblocks(X, block_size):
    """

    Naive function to split input square matrix into blocks defined by
    the given block size
    Inputs:
        X           :   Input matrix to be divided into blocks
        block_size  :   Block size
    Returns:
        list        :   A list of numpy arrays, each array is a block of shape
                        defined by input block_size

    """

    rows, cols = X.shape

    nrows = block_size[0]
    ncols = block_size[1]
    temp = (X.reshape(cols//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))
    list = [x for x in temp]
    return list


def buildback(list, block_size, og_size):
    """

    Inverse of function 'makeblocks'.
    Reorganizes processed block-wise data into a matrix of shape equal to
    the original matrix given to 'makeblocks'.
    Inputs:
        list            :   list of blocks from 'makeblock'
        block_size      :   shape of the blocks in the list
        og_size         :   Original shape of input array, the size which
                            the returned array will be reshaped to.
    Returns:
        data            :   data array reshaped to desired shape

    Note: block_size passed explicitly to avoid confusion, function can be
    rewritten to infer the block_size by checking list[i].shape

    """

    nrows = block_size[0]
    ncols = block_size[1]
    nblocks = og_size[0]//nrows
    arr = np.array(list).reshape(og_size[0]//nrows, og_size[1]//ncols, nrows, ncols)
    arr = arr.swapaxes(1, 2)
    data = arr.reshape(og_size[0], og_size[1])

    return data


def blockSVT(Z, block_size, lambd):
    """

    Computes block-wise SVT over blocks of Z, defined by block_size
    Inputs:
        Z           :       Input matrix
        block_size  :       A tuple, square block_size (b, b)
        lambd       :       Lambda - threshold
    Returns:
        Z           :       Block-wise thresholded Z

    """
    eps = np.finfo(np.float64).eps # machine epsilon
    doBlockSVT = lambda X: SVT(X, lambd)

    if block_size[0] == Z.size:
        t = np.norm(Z.flatten(), 2)
        Z = np.dot(SoftThresh(t, lambd), Z) / (t + eps)

    else:
        Z_shape = Z.shape
        data = makeblocks(Z, block_size)
        outdata = [doBlockSVT(x) for x in data]
        Z = buildback(outdata, block_size, Z_shape)

    return Z
