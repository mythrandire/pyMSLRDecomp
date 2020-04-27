import numpy as np
from numpy.linalg import svd

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
        ZZ = np.multiply(Z.T, Z)
    else:
        ZZ = np.multiply(Z, Z.T)

    if max(np.sum(abs(Z), 1)) < lambd/2:
        Z = np.multiply(Z, 0)
    else:
        U, S, V, _ = svd(Z) # verify if this works
        Z = np.multiply(np.multiply(U, np.diag(SoftThresh(np.diag(S), lambd))),
                        V.T)

    return Z
