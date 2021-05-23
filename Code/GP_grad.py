from typing import overload
import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import inv
from GP import GP

def sq_dist(x1, x2): # x1: N*d, x2: M*d  --> N*d
    if x1.shape[1] != x2.shape[1]: # 2d
        x1 = np.reshape(x1, (-1, x2.shape[1]))
    return np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)

def diff(x1, x2): # x1: N*1, x2: M*1
    return x1 - np.squeeze(x2)

class GP_grad(GP):
    def __init__(self, SearchSpace, Noise=False, noise_delta=1e-4, p=0, verbose=0):
        super().__init__(SearchSpace, Noise, noise_delta, verbose)
        self.p = p  # partial derivative along which the grad GP (0 to N-1)

    def set_p(self, p=0):
        self.p = p

    def K11(self, Xtest, hyper=None): # RBF Kernel
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]

        Xtest_p = np.reshape(Xtest[:, self.p], (-1, 1))

        sqdist = sq_dist(Xtest, Xtest)
        sqdist_pp = sq_dist(Xtest_p, Xtest_p)
        return (variance/(lengthscale**2)) * (1 - sqdist_pp / (lengthscale**2)) * np.exp(-0.5 * sqdist / (lengthscale**2))


    def K01(self, Xtest, hyper=None):
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]

        Xtest_p = Xtest[:, self.p]
        X_p = np.reshape(self.X[:, self.p], (-1, 1))

        sqdist = sq_dist(self.X, Xtest)
        diff = X_p - Xtest_p
        K_01 = (variance/(lengthscale**2)) * diff * np.exp(-0.5 * sqdist / (lengthscale**2))
        return K_01

    def joint_MVN(self, Xtest): # give the joint distribution of gp and derivative
        N = self.X.shape[0]
        M = Xtest.shape[0]

        if len(Xtest.shape) == 1:  # 1d
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))

        K = self.cov_RBF(self.X, self.X)
        K_11 = self.K11(Xtest)
        K_01 = self.K01(Xtest)
        K_10 = K_01.T
        up = np.hstack([K, K_01])
        down = np.hstack([K_10, K_11])
        joint_cov = np.vstack([up, down])
        return np.zeros((M+N, 1)), joint_cov

    def prior_grad(self, Xtest):
        M = Xtest.shape[0]
        if len(Xtest.shape) == 1:  # 1d
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))
        K_11 = self.K11(Xtest, Xtest)
        return np.zeros((M, 1)), K_11

    def posterior_grad(self, Xtest):
        """
        Xtest: the testing points  [M*d]
        Returns: posterior mean, posterior var of grad GP
        """
        if len(Xtest.shape) == 1:  # 1d
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))

        if Xtest.shape[1] != self.X.shape[1]:  # different dimension
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))


        self.alpha = self.fit()
        # assert self.K != None
        # assert self.L != None
        K_01 = self.K01(Xtest)

        meanPost = np.reshape(np.dot(K_01.T, self.alpha), (-1, 1))
        v = np.linalg.solve(self.L, K_01)
        covPost = self.K11(Xtest) - np.dot(v.T, v)
        # var = np.reshape(np.diag(var), (-1, 1))
        return meanPost, covPost
