import numpy as np


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X):
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        mu = np.mean(X, axis = 0, keepdims = True)
        X = X - mu
        U, S, V = np.linalg.svd(X, full_matrices = False)
        self.U = U
        self.S = S
        self.V = V
        # raise NotImplementedError

    def transform(self, data, K=2):
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        self.fit(data)
        X_new = np.dot(data, self.V.T[:, :K])
        return X_new
        # raise NotImplementedError

    def transform_rv(self, data, retained_variance=0.99):
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        self.fit(data)
        var = np.cumsum(self.S ** 2)
        var = var / var[-1]

        N = data.shape[0]

        for i in range(N):
            if var[i] > retained_variance:
                break

        X_new = np.dot(data, self.V.T[:, :i + 1])
        return X_new

        # raise NotImplementedError

    def get_V(self):
        """ Getter function for value of V """

        return self.V