import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.image as mgimg
from matplotlib import animation

def animate(iter):

    fig = plt.figure()
    # initiate an empty  list of "plotted" images
    myimages = []

def liveplot(X, iter):
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
    plt.imshow(X_new), plt.title('Decomposition: Iteration %i' %iter)
    plt.draw()
    plt.pause(0.01)
