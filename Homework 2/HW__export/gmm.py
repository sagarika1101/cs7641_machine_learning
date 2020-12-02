from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio



# Set random seed so output is all same
np.random.seed(1)



class GMM(object):
    def __init__(self, X, K, max_iters = 100): # No need to change
        """
        Args: 
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        
        self.N = self.points.shape[0]        #number of observations
        self.D = self.points.shape[1]        #number of features
        self.K = K                           #number of components/clusters

    #Helper function for you to implement
    def softmax(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        # max of each row in logit
        max_logit = np.max(logit, axis=1)
        # subtract max of each row from logit
        logit -= max_logit[:, None]
        # initialize prob
        prob = np.exp(logit)
        prob /= np.sum(np.exp(logit), axis=1)[:, None]
        return prob
        # raise NotImplementedError

    def logsumexp(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        # max of each row in logit
        max_logit = np.max(logit, axis=1)
        # subtract max of each row from logit
        logit -= max_logit[:, None]
        # initialize s
        s = np.log(np.sum(np.exp(logit), axis=1)[:, None] + 1e-32)
        # add max back to s
        s += max_logit[:, None]
        return s
        # raise NotImplementedError

    #for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i): #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array (or array of lenth D), the center for the ith gaussian.
            sigma_i: 1xDxD 3-D numpy array (or DxD 2-D numpy array), the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array (or array of length N), the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        
        raise NotImplementedError
    
    #for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array (or array of lenth D), the center for the ith gaussian.
            sigma_i: 1xDxD 3-D numpy array (or DxD 2-D numpy array), the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array (or array of length N), the probability distribution of N data for the ith gaussian
         
        Hint: 
            np.linalg.det() and np.linalg.inv() should be handy.
        """
        N, D = logits.shape[0], logits.shape[1]
        mu_i.reshape((D, 1))
        # term1
        # adding very small value to sigma to avoid singular matrix error
        t1 = np.dot((logits - mu_i.T[None, :]), np.linalg.inv(sigma_i+0.00001)).T
        # term2
        t2 = (logits - mu_i.T[None, :]).T
        # element-wise multiplication
        t3 = np.exp(np.sum(np.multiply(t1, t2), axis=0) * (-0.5))
        # term 4
        t4 = np.power(np.linalg.det(sigma_i), -0.5)
        # term 5
        t5 = np.power(np.power((2 * np.pi), D / 2), -1)
        # putting it all together
        pdf = t5 * t4 * t3
        pdf.reshape((1, N))
        return pdf
        # raise NotImplementedError
    
    
    def _init_components(self, **kwargs): # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
                You will have KxDxD numpy array for full covariance matrix case
        """
        N, D, K = self.N, self.D, self.K
        points = self.points

        # initialize pi as same probability for each class
        pi = np.array([1 / K] * K)

        # randomly select K observations as mean vectors
        indices = np.random.choice(points.shape[0], size=K, replace=False)
        # form the numpy array with these indices
        mu = points[indices, :]

        # sigma is np.eye for each K
        sigma = np.array([np.eye(D)] * K)

        return pi, mu, sigma
        # raise NotImplementedError

    
    def _ll_joint(self, pi, mu, sigma, **kwargs): # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        N, D, K = self.N, self.D, self.K
        logits = self.points
        # initialize ll
        ll = np.zeros((N, K))

        # term1
        t1 = np.log(pi + 1e-32)
        # add t1 to ll
        ll = ll + t1

        # term2
        for i in range(K):
            mu_i = mu[i, :]
            sigma_i = sigma[i, :, :]
            pdf = self.multinormalPDF(logits, mu_i, sigma_i)
            pdf = np.nan_to_num(pdf)
            ll[:, i] += np.log(pdf + 1e-32)

        return ll

        # raise NotImplementedError

    def _E_step(self, pi, mu, sigma, **kwargs): # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: 
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        logits = self.points
        N, D, K = self.N, self.D, self.K

        # initialize tau
        tau = np.zeros((N, K))

        for i in range(K):
            mu_i = mu[i, :]
            sigma_i = sigma[i, :, :]
            pdf = self.multinormalPDF(logits, mu_i, sigma_i)
            if np.isnan(pdf).any():
                print('pdf has 0')
            tau[:, i] = np.dot(pi[i], pdf).T

        # term2, summation
        t2 = np.sum(tau, axis=1)
        tau = tau / t2[:, None]

        return tau

        # raise NotImplementedError

    def _M_step(self, gamma, **kwargs): # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Hint:  
            There are formulas in the slide and in the above description box.
        """
        N_k = np.sum(gamma, axis=0)
        pi, mu, sigma = self._init_components()
        logits = self.points
        N, D, K = self.N, self.D, self.K

        # mu_k
        mu_new = np.dot(gamma.T, logits) / N_k.reshape(-1, 1)

        # sigma_k
        sigma_new = np.zeros((K, D, D))
        for i in range(K):
            # term1
            t1 = (logits - mu_new[i]).T
            # term2, element-wise multiplication
            t2 = gamma[:, i] * t1
            #             t2 = np.multiply(gamma[:, i], t1)
            # term3, dot product
            t3 = np.dot(t2, (logits - mu_new[i]))
            sigma_new[i] = t3 / N_k[i]

            # pi_k
        pi_new = np.zeros((K))
        pi_new = N_k / N
        pi_new.reshape((K))

        return pi_new, mu_new, sigma_new

        # raise NotImplementedError
    
    
    def __call__(self, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        
        Hint: 
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters. 
        """
        pi, mu, sigma = self._init_components(**kwargs)
        if np.isnan(pi).any():
            print('pi has 0')
        if np.isnan(mu).any():
            print('mu has 0')
        if np.isnan(sigma).any():
            print('sigma has 0')
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)
            gamma = np.nan_to_num(gamma)
            if np.isnan(gamma).any():
                print('gamma has 0')

            # M-step
            pi, mu, sigma = self._M_step(gamma)
            pi = np.nan_to_num(pi)
            mu = np.nan_to_num(mu)
            sigma = np.nan_to_num(sigma)
            if np.isnan(pi).any():
                print('pi_new has 0')
            if np.isnan(mu).any():
                print('mu_new has 0')
            if np.isnan(sigma).any():
                print('sigma_new has 0')

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma)
            joint_ll = np.nan_to_num(joint_ll)
            if np.isnan(joint_ll).any():
                print('joint_ll has 0')
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
