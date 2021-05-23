import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import block_diag

class GP(object):
    def __init__(self, SearchSpace, Noise=False, noise_delta=1e-4, verbose=0):  # noise_delta=1e-8
        self.noise_delta = noise_delta
        self.noise_upperbound = noise_delta
        self.SearchSpace = SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler = scaler
        self.verbose = verbose
        self.dim = SearchSpace.shape[0]
        self.hyper = {}
        self.hyper["var"] = 1
        self.hyper["lengthscale"] = 1
        # set data to be None
        self.X = None
        self.y = None

        self.fitted = False

        if Noise == True:
            self.noise_delta = noise_delta
        else:
            self.noise_delta = 1e-4

    def set_data(self, X, y): # X: input 2d array [N*d], y: output 2d array [N*1]
        # self.X = self.Xscaler.transform(X) # normalised data [0-1] in each column
        self.X = X
        self.y = np.reshape(y, (self.X.shape[0], 1)) # the standardised output N(0,1)
        self.fitted = False

    def add_data(self, X, y): # X [N*d], y [N*1]
        assert len(y.shape) != 0
        self.X = np.vstack((self.X, np.reshape(X, (X.shape[0], -1))))
        self.y = np.vstack((self.y, np.reshape(y, (y.shape[0], -1))))
        self.fitted = False

    def set_hyper(self, lengthscale, variance):
        self.hyper["lengthscale"] = lengthscale
        self.hyper["var"] = variance

    def get_hyper(self):
        return self.hyper

    def cov_RBF(self, x1, x2, hyper=None): # RBF Kernel
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]

        if x1.shape[1] != x2.shape[1]: # 2d
            x1 = np.reshape(x1, (-1, x2.shape[1]))

        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
        return variance * np.exp(-0.5 * sqdist / np.square(lengthscale))

    def log_lik(self, hyper_values):
        # min the -ve loglk for the estimation of ls and var
        hyper = {}
        hyper["var"] = hyper_values[1]
        hyper["lengthscale"] = hyper_values[0]

        KK_x_x = self.cov_RBF(self.X, self.X, hyper) + np.eye(len(self.X)) * self.noise_delta
        if np.isnan(KK_x_x).any():  # NaN
            print("NaN in KK_x_x")
        
        try:
            L = scipy.linalg.cholesky(KK_x_x, lower=True)
            temp = np.linalg.solve(L, self.y)
            alpha = np.linalg.solve(L.T, temp) # found the alpha for given hyper parameters
        except:
            # print(hyper)
            # print(KK_x_x)
            raise ValueError('bad')

        log_lik = -1/2*np.dot(self.y.T, alpha) - np.sum(np.log(np.diag(L))) - 0.5*len(self.y)*np.log(2*np.pi)
        return np.asscalar(log_lik)

    def optimize(self): # Optimise the GP kernel hyperparameters
        opts = {"maxiter": 200, "maxfun": 200, "disp": False}
        bounds = np.asarray([[1e-2, 1], [0.05, 1.5]])  # bounds on Lenghtscale and kernal Variance

        W = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10, 2))
        loglik = np.array([])

        for x in W:
            loglik = np.append(loglik, self.log_lik(hyper_values=x))
        
        x0 = W[np.argmax(loglik)]
        Res = minimize(lambda x: -self.log_lik(hyper_values=x), 
                                x0,
                                bounds=bounds, 
                                method="L-BFGS-B", 
                                options=opts) # L-BFGS-B

        if self.verbose:
            print("estimated lengthscale and variance = ", Res.x)
        ls, var = Res.x
        # print('ls', ls)
        # print('var', var)
        self.set_hyper(ls, var) # keep the original paramters
        return Res.x
    

    def prior(self, Xtest, isOriScale=False):
        """
        Xtest: the testing points  [M*d]
        Returns: prior mean, prior var given our hyperparameters
        """
        M = Xtest.shape[0]
        if isOriScale:
            Xtest = self.Xscaler.transform(Xtest)

        if len(Xtest.shape) == 1:  # 1d
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))

        if Xtest.shape[1] != self.X.shape[1]:  # mismatched dimension
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1])) # reshape into [M*d]

        covPrior = self.cov_RBF(Xtest, Xtest, self.get_hyper())
        meanPrior = np.zeros((M, 1))
        return meanPrior, covPrior

    def fit(self): # find alpha with self.hyper
        self.K = self.cov_RBF(self.X, self.X, self.hyper) + np.eye(len(self.X)) * self.noise_delta
        if np.isnan(self.K).any():  # NaN
            print("NaN in K")
        self.L = scipy.linalg.cholesky(self.K, lower=True)
        # print(self.L.shape)
        # print(self.y.shape)
        temp = np.linalg.solve(self.L, self.y)
        self.alpha = np.linalg.solve(self.L.T, temp) # algorithm 15.1
        self.fitted = True
        return self.alpha

    def posterior(self, Xtest, isOriScale=False):
        """
        Xtest: the testing points  [M*d]
        Returns: posteior mean, posteior var
        """
        if isOriScale:
            Xtest = self.Xscaler.transform(Xtest)

        if len(Xtest.shape) == 1:  # 1d
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))

        if Xtest.shape[1] != self.X.shape[1]:  # mismatched dimension
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1])) # reshape into [M*d]

        KK_xTest_xTest = self.cov_RBF(Xtest, Xtest, self.hyper)
        KK_x_xTest = self.cov_RBF(self.X, Xtest, self.hyper)

        meanPost = np.reshape(np.dot(KK_x_xTest.T, self.alpha), (-1, 1))
        v = np.linalg.solve(self.L, KK_x_xTest)
        covPost = KK_xTest_xTest - np.dot(v.T, v)
        # var = np.reshape(np.diag(var), (-1, 1))
        return meanPost, covPost

