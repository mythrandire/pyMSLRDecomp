import numpy as np
from sklearn.feature_extraction import image as I
from skimage.util.shape import view_as_blocks
from SVT import SVT
from SoftThresh import SoftThresh

def makeblocks(X, block_size):
    """
    Naive function to split input square matrix into blocks defined by
    the given block size
    """


    rows, cols = X.shape

    nrows = block_size[0]
    ncols = block_size[1]
    temp = (X.reshape(cols//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))
    list = [x for x in temp]
    return list



def blockSVT(Z, block_size, lambd):
    """
    Computes block-wise SVT over blocks of Z, defined by block_size
    Inputs:
        Z           :       Input matrix
        block_size  :       A tuple, square block_size (b, b)
        lambd       :       Lambda - threshold
    Returns:
        Z           :       Block-wise thresholded
    """
    eps = np.finfo(float).eps
    doBlockSVT = lambda X: SVT(X, lambd)
    if block_size[0] == Z.size:
        t = np.norm(Z.flatten(), 2)
        Z = np.multiply(SoftThresh(t, lambd), Z) / (t + eps)
    else:
        # consider list comprehension after manual block creation to simulate
        # blockproc?
        # output_list = [function(x) for x in input_list]
        #
        # data = I.extract_patches_2d(Z, block_size, int(Z.shape[0]/block_size[0]))
        data = makeblocks(Z, block_size)
        print("block data shape: ", len(data))
        #print(data)
        outdata = [doBlockSVT(x) for x in data]
        #temp = doBlockSVT(data)
        print("Size of BSVT output: ", len(outdata))



    return Z
