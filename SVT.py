import numpy as np
from numpy.linalg import svd
from SoftThresh import SoftThresh

def SVT(Z, lambd):
    """
    Performs Singular Value Thresholding on the input matrix Z.
    Inputs:
        Z       :   Target matrix
        lambd   :   threshold
    Returns:
        Z       : Z, singular value-thresholded
    """

    if (Z.shape[0] > Z.shape[1]):
        ZZ = np.dot(Z.T, Z)
    else:
        ZZ = np.dot(Z, Z.T)

    if np.max(np.sum(abs(ZZ), 1)) < lambd/2:
        Z = np.dot(Z, 0)
    else:
        U, S, V = svd(Z) # verify if this works
        Z = U * SoftThresh(S, lambd) * V.T

    return Z
