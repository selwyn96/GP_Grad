import numpy as np
from scipy.stats import norm
import torch
import gpytorch
import scipy.optimize as spo
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.stats import norm
import sobol_seq
from torch.quasirandom import SobolEngine
from utils import unique_rows
from Gaussian import GaussianProcess

class methods(object):
    # Contain additional acq functions (slightly more complex ones)

    def __init__(self, acq_name,bounds,model,Y):

            self.acq_name = acq_name
            self.bounds=bounds
            self.model=model
            self.dim = len(self.bounds)
            self.sobol = SobolEngine(self.dim, scramble=True)
            self.lb=np.array(self.bounds)[:,0]
            self.ub=np.array(self.bounds)[:,1]
            self.center=(self.lb +self.ub)/2 # The center of the unit domain, used for Satisfying TS
            self.n_samples=1000
            self.Y=Y

    
    def method_val(self):
  
        if self.acq_name == 'random' :
            return self._random()
        
        if self.acq_name == 'TS' :
            return self._TS()

        if self.acq_name == 'MES' :
            return self._MES()
    
    # randomly samples a point in the domain
    def _random(self):
        x= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 1, self.bounds.shape[0]))[0]
        return x
        
    
    # Thompsons sampling
    # Each sampled function is discrete (10000 points)
    def _TS(self):
        coordinates= self.sobol.draw(1000).cpu().numpy() * (self.ub - self.lb) + self.lb
        X_tries=coordinates
        samples = self.model.sample(X_tries,size=1)
        index=np.argmax(samples)
        return(X_tries[index])
    
    # Gumble sampling for sampling max values in MES
    def sample_maxes_G(self):
        x_grid = sobol_seq.i4_sobol_generate(self.dim, 100)
        mu,var = self.model.predict(x_grid)
        std = np.sqrt(var)

        def cdf_approx(z):
            z = np.atleast_1d(z)
            ret_val = np.zeros(z.shape)
            for i, zi in enumerate(z):
                ret_val[i] = np.prod(norm.cdf((zi - mu) / std))
            return ret_val

        lower = np.max(self.Y)
        upper = np.max(mu + 5*std)
        if cdf_approx(upper) <= 0.75:
            upper += 1

        grid = np.linspace(lower, upper, 100)

        cdf_grid = cdf_approx(grid)
        r1, r2 = 0.25, 0.75

        y1 = grid[np.argmax(cdf_grid >= r1)]
        y2 = grid[np.argmax(cdf_grid >= r2)]

        beta = (y1 - y2) / (np.log(-np.log(r2)) - np.log(-np.log(r1)))
        alpha = y1 + (beta * np.log(-np.log(r1)))
        maxes = alpha - beta*np.log(-np.log(np.random.rand(self.n_samples,)))
        return maxes
     
    # Thompsons sampling for finding max values in MES
    def sample_maxes_T(self):
        x_sobol = sobol_seq.i4_sobol_generate(self.dim, 1000)
        X_tries=x_sobol
        samples = samples = self.model.sample(X_tries,size=100)
        samples=samples.detach().cpu().numpy()
        maxs = np.max(samples, axis=0)
        percentiles = np.linspace(50, 95, self.n_samples)
        reduced_maxes = np.percentile(maxs, percentiles)
        print(reduced_maxes)
        return reduced_maxes
    
    # The MES acquistion function
    def acq(self,x,y_maxes):
        x = np.atleast_2d(x)
        mu,var = self.model.predict(x)
        std=np.sqrt(var)
        mu=mu.flatten()
        std=std.flatten()
        gamma_maxes = (y_maxes - mu) / std[:, None]
        tmp = 0.5 * gamma_maxes * norm.pdf(gamma_maxes) / norm.cdf(gamma_maxes) - \
            np.log(norm.cdf(gamma_maxes))
        mes = np.mean(tmp, axis=1, keepdims=True)
        mes= np.nan_to_num(mes)
        return mes
    
    # Optimizing the MES function
    def _MES(self):
        y_maxes = self.sample_maxes_G()
        x_tries= self.sobol.draw(100).cpu().numpy() * (self.ub - self.lb) + self.lb
        ys=np.array([])
        for x_try in x_tries:
            saved = self.acq(x_try.reshape(1, -1),y_maxes)
            ys=np.append(ys,saved)
        x_max = x_tries[np.random.choice(np.where(ys == ys.max())[0])]
        max_acq = ys.max()
        # Explore the parameter space more throughly
        x_seeds = self.sobol.draw(10).cpu().numpy() * (self.ub - self.lb) + self.lb
        for x_try in x_seeds:
            res = minimize(lambda x: -self.acq(x.reshape(1, -1),y_maxes),x_try.reshape(1, -1),bounds=self.bounds,method="L-BFGS-B")
            if not res.success:
                continue
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        return np.clip(x_max, self.bounds[:, 0],self.bounds[:, 1])
    

