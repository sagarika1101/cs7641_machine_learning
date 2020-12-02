import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        rmse = np.sqrt(np.mean((pred - label) ** 2))
        return rmse
        # raise NotImplementedError

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        N = x.shape[0]
        feat = np.ones((N))
        for i in range(1, degree + 1):
            x_term = np.power(x, i)
            temp = np.column_stack((feat, x_term))
            feat = temp
        return feat
        # raise NotImplementedError

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        prediction = np.dot(xtest, weight)
        return prediction
        # raise NotImplementedError

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        # term1
        t1 = np.linalg.pinv(np.dot(xtrain.T, xtrain))
        # term2
        t2 = np.dot(t1, xtrain.T)
        weight = np.dot(t2, ytrain)
        return weight
        # raise NotImplementedError

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape[0], xtrain.shape[1]
        weight = np.zeros((D, 1))
        for i in range(epochs):
            temp = weight + learning_rate * (np.dot(xtrain.T, (ytrain - np.dot(xtrain, weight)))) / N
            weight = temp
        return weight
        # raise NotImplementedError

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape[0], xtrain.shape[1]
        weight = np.zeros((D, 1))
        for i in range(epochs):
            t1 = ytrain - np.dot(xtrain, weight)
            idx = i % N
            t2 = np.dot(xtrain[idx, :].T.reshape((D, 1)), t1[idx].reshape((1, 1)))
            temp = weight + learning_rate * t2
            weight = temp
        return weight
        # raise NotImplementedError

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        N, D = xtrain.shape[0], xtrain.shape[1]
        ide = np.identity(D)
        ide[0][0] = 0.0
        # term1
        t1 = np.linalg.pinv(np.dot(xtrain.T, xtrain) + c_lambda * ide)
        # term2
        t2 = np.dot(t1, xtrain.T)
        weight = np.dot(t2, ytrain)
        return weight
        # raise NotImplementedError

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape[0], xtrain.shape[1]
        weight = np.zeros((D, 1))
        for i in range(epochs):
            #term 1
            t1 = np.dot(xtrain.T, (ytrain - np.dot(xtrain, weight)))
            temp = weight + learning_rate * (t1 + c_lambda * weight) / N
            weight = temp
        return weight
        # raise NotImplementedError

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape[0], xtrain.shape[1]
        weight = np.zeros((D, 1))
        for i in range(epochs):
            t1 = ytrain - np.dot(xtrain, weight)
            idx = i % N
            t2 = np.dot(xtrain[idx, :].T.reshape((D, 1)), t1[idx].reshape((1, 1)))
            temp = weight + learning_rate * (t2 + c_lambda * weight)
            weight = temp
        return weight
        # raise NotImplementedError

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        N, D = X.shape[0], X.shape[1]
        size_of_fold = int(N / kfold)
        meanErrors = 0.0
        for i in range(kfold):
            # train model on every fold except ith
            xtrain1 = X[:i * size_of_fold, :]
            xtrain2 = X[(i + 1) * size_of_fold:, :]
            xtrain = np.concatenate((xtrain1, xtrain2))
            ytrain1 = y[:i * size_of_fold]
            ytrain2 = y[(i + 1) * size_of_fold:]
            ytrain = np.concatenate((ytrain1, ytrain2))
            weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)

            # compute error on ith fold
            xtest = X[i * size_of_fold:(i + 1) * size_of_fold, :]
            ytest = y[i * size_of_fold:(i + 1) * size_of_fold]
            pred = self.predict(xtest, weight)
            error = self.rmse(pred, ytest)

            # add to error
            meanErrors += error

        meanErrors /= kfold
        return meanErrors
        # raise NotImplementedError