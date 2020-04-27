import numpy as np

def randshift(in):
    """
    Description
    """
    s = in.shape
    r = np.zeros_like()
    # use numpy.roll
