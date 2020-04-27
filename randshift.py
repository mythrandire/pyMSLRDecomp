import numpy as np

def randshift(in):
    """
    Description
    """
    s = in.shape
    r = np.zeros_like(1, len(s))

    for  i in range(0, len(s)):
        r[i] = np.random.randint(s[i])

    out = np.roll(in, r[i])

    return out
    # use numpy.roll
