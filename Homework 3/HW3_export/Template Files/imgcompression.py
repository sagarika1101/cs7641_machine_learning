from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N (*3 for color images)
            S: min(N, D) * 1 (* 3 for color images)
            V: D * D (* 3 for color images)
        """
        # BnW image
        if len(X.shape) == 2:
            U, S, V = np.linalg.svd(X)
        else:
            # colour image
            N, D = X.shape[0], X.shape[1]
            min_ND = min(N, D)

            # channel0
            U0, S0, V0 = np.linalg.svd(X[:, :, 0])
            S0 = S0.reshape((min_ND, 1))

            # channel1
            U1, S1, V1 = np.linalg.svd(X[:, :, 1])
            S1 = S1.reshape((min_ND, 1))

            # channel2
            U2, S2, V2 = np.linalg.svd(X[:, :, 2])
            S2 = S2.reshape((min_ND, 1))

            #combining
            U = np.array([U0, U1, U2])
            U = U.transpose(1, 2, 0)

            S = np.concatenate((S0, S1, S2), axis=1)

            V = np.array([V0, V1, V2])
            V = V.transpose(1, 2, 0)
        return U, S, V

    # raise NotImplementedError


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U.shape) == 2:
            # BnW image
            # term1
            t1 = np.matmul(U[:, :k], np.diag(S)[:k, :k])
            # term2
            Xrebuild = np.matmul(t1, V[:k, :])
        else:
            # colour image
            N = U.shape[0]
            D = V.shape[0]

            # channel 0
            U0 = U[:, :, 0]
            S0 = S[:, 0]
            V0 = V[:, :, 0]
            # term1
            t1 = np.matmul(U0[:, :k], np.diag(S0)[:k, :k])
            # term2
            Xrebuild0 = np.matmul(t1, V0[:k, :])

            # channel 1
            U1 = U[:, :, 1]
            S1 = S[:, 1]
            V1 = V[:, :, 1]
            # term1
            t1 = np.matmul(U1[:, :k], np.diag(S1)[:k, :k])
            # term2
            Xrebuild1 = np.matmul(t1, V1[:k, :])

            # channel 2
            U2 = U[:, :, 2]
            S2 = S[:, 2]
            V2 = V[:, :, 2]
            # term1
            t1 = np.matmul(U2[:, :k], np.diag(S2)[:k, :k])
            # term2
            Xrebuild2 = np.matmul(t1, V2[:k, :])

            # combining
            Xrebuild = np.array([Xrebuild0, Xrebuild1, Xrebuild2])
            Xrebuild = Xrebuild.transpose(1, 2, 0)

        return Xrebuild

        # raise NotImplementedError

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        if len(X.shape) == 2:
            # BnW image
            num_orig = X.shape[0] * X.shape[1]
            num_compress = k * (1 + X.shape[0] + X.shape[1])
        else:
            # colour image
            num_orig = X.shape[0] * X.shape[1] * X.shape[2]
            num_compress = k * (1 + X.shape[0] + X.shape[1]) * X.shape[2]

        compression_ratio = num_compress * 1.0 / num_orig
        return compression_ratio
        # raise NotImplementedError

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        S2 = S ** 2
        if len(S.shape) == 1:
            # BnW image
            recovered_var = np.sum(S2[:k]) / np.sum(S2)
        else:
            # colour image
            # channel0
            recovered_var0 = np.sum(S2[:k, 0]) / np.sum(S2[:, 0])
            # channel1
            recovered_var1 = np.sum(S2[:k, 1]) / np.sum(S2[:, 1])
            # channel2
            recovered_var2 = np.sum(S2[:k, 2]) / np.sum(S2[:, 2])
            recovered_var = [recovered_var0, recovered_var1, recovered_var2]

        return recovered_var
        # raise NotImplementedError