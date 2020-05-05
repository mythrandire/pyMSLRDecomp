import numpy as np

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
