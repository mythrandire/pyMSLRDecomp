import numpy as np

def signum(y):
    """
    Custom signum function to address discrepenency in how matlab and numpy
    handle extremely small floats

    """
    eps = np.finfo(np.float64).eps # machine epsilon
    y[y >= -eps] = 1.0
    y[y < -eps] = 0.0

    return y

def SoftThresh(y, t):
    """
    Returns Soft-Thresholded version of noisy input 'y'. The threshold is
    defined by 't'.
    Inputs:
        y   :   Noisy Input
        t   :   Theshold
    Returns:
        x   :   signum(y)(|y| - t)_{+}
    """
    res = np.abs(y) - t
    res = (res + np.abs(res))/2
    x = np.multiply(np.sign(y), res)
    return x
