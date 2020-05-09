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

    Notes: np.linalg.svd returns USV.T, so no need to transpose when
    recombining.
    """

    ZZ = np.dot(Z, Z.T)

    if np.max(np.sum(abs(ZZ), 1)) < lambd**2:
        Z = np.dot(Z, 0)
        #print('if2')
    else:
        U, S, V = svd(Z) # verify if this works
        Z = np.matmul(np.matmul(U, SoftThresh(np.diag(S), lambd)), V)
        #print('else2')

    return Z
