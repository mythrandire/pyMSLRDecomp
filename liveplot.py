"""
liveplot

Multi-scale Low Rank Image Decomposition in Python

Author: Abhishek Bhan

Function to compute Singular Value Threshold.

"""


import numpy as np
import matplotlib.pyplot as plt


def liveplot(X, iter):
    """

    Function to draw and display decomposition in real time
    Inputs:
        X      :    3D array of decomposed slices
        iter   :    iteration count of ADMM solver
    Returns:

    """

    X_new = np.concatenate((X[:,:,0], X[:,:,1], X[:,:,2], X[:,:,3]), axis=1)
    plt.imshow(X_new), plt.title('Decomposition: Iteration %i' %iter)
    plt.draw()
    plt.pause(0.001)
