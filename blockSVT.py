import numpy as np
from sklearn.feature_extraction import image as I
from skimage.util.shape import view_as_blocks
import SVT
import SoftThresh

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
        # data = I.extract_patches_2d(Z, block_size, Z.shape[0]/block_size[0])
        data = view_as_blocks(Z, block_size)
        Z = doBlockSVT(data).reshape(Z.shape)

    return Z
