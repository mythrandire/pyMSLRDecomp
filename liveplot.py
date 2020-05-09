import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def liveplot(X):
    """
    Description
    Inputs:
        img     :
        range   :
        shape   :
    Returns:

    """
    #r, c, d = X.shape
    #nrows, ncols = grid_shape

    X_new = np.concatenate((X[:,:,0], X[:,:,1], X[:,:,2], X[:,:,3]), axis=1)
    # X_new = np.concatenate((X[:,:,0], X[:,:,1]), axis=1)
    #norm = Normalize(vmin=0, vmax=255, clip=False)
    plt.imshow(X_new), plt.title('Decomposition')
    plt.show()
