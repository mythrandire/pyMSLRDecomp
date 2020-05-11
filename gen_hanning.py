import numpy as np

def gen_hanning(FOV, block_sizes, nblocks, sigma):
    """
    function to generate synthetic images using the hanning window
    """
    levels = len(block_sizes)
    X_decom = np.zeros((FOV+ (levels,)))
    # np.random.seed(5) # set fixed seed for (almost) reproducible test image
    for l in range(levels):

        if block_sizes[0][0] == np.prod(np.asarray(FOV)):
            X_decom[:,:,l] = np.random.randn(FOV)*sigma

        else:

            for n in range(nblocks[l]):
                u = np.zeros(FOV[0])
                v = np.zeros(FOV[1])

                bs1, bs2 = block_sizes[l]
                pos = np.array(
                                [(np.random.randint(0, FOV[0]//bs1, 2) - 1)*bs1,
                                 (np.random.randint(0, FOV[1]//bs2, 2) - 1)*bs2
                                ]
                              ).flatten()
                u[np.array(np.arange(0, bs1)+pos[0])] = np.hanning(bs1+2)[1:-1]
                v[np.array(np.arange(0, bs2)+pos[1])] = np.hanning(bs2+2)[1:-1]

                X_decom[:,:,l] += np.multiply(u.reshape(FOV[0], 1),
                                              v.reshape(1, FOV[1]))

    X = np.sum(X_decom, axis=2)/np.sqrt(levels)

    return X, X_decom
