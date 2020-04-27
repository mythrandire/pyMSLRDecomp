import numpy as np
"""
def gen_hanning(FOV, block_sizes, n_blocks, sigma):
    """
    Description
    """
    levels = len(block_sizes)
    X_decom = np.zeros((FOV, levels))

    for i in range(levels):

        if block_sizes(i, 1) == np.prod(FOV):
            X_decom[:,:,0] = np.random.randn(FOV[0], FOV[1]) * sigma
        else:

            for j in range(n_blocks[i]):
                u = np.zeros((FOV[0], 1))
                v = np.zeros((FOV[1], 1))

                block_size1 = block_sizes[i][0]
                block_size2 = block_sizes[i][1]

                pos = np.random.randint(np.floor(FOV[0]/block_size1), 1, 2)

"""
