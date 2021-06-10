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
import math 

class methods(object):
    # Contain additional acq functions (slightly more complex ones)

    def __init__(self, acq_name,bounds,model,model_1,obj,Y,X,Count):

            self.acq_name = acq_name
            self.bounds=bounds
            self.model=model
            self.model_1=model_1
            self.dim = len(self.bounds)
            self.sobol = SobolEngine(self.dim, scramble=True)
            self.lb=np.array(self.bounds)[:,0]
            self.ub=np.array(self.bounds)[:,1]
            self.center=(self.lb +self.ub)/2 # The center of the unit domain, used for Satisfying TS
            self.n_samples=1000
            self.Y=Y
            self.present_val=X
            self.count=Count
            self.obj=obj

    
    def method_val(self):
  
        if self.acq_name == 'random' :
            return self._random()
        
        if self.acq_name == 'TS' :
            return self._TS()

        if self.acq_name == 'MES' :
            return self._MES()

        if self.acq_name == 'GD' :
            return self._GD()
    
    # randomly samples a point in the domain
    def _random(self):
        x= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 1, self.bounds.shape[0]))[0]
        print(x)
        return x
    
    # Gradient descent algo

    def _GD(self):
        LR=0.2
      #  LR=0.01+0.5*(0.1-0.01)*(1+math.cos((self.count/20)*math.pi))
        print(LR)
        if(self.count==0):
                starting_point=self.find_initial_point()
        else:
                starting_point=self.present_val

        if self.dim==1:
         #   mean, var = self.model.predict(starting_point+0.9*self.obj.return_moment())
            mean = self.model.sample(starting_point+0.9*self.obj.return_moment(),size=1).flatten()
            new_momentum=0.9*self.obj.return_moment()+LR*mean
            new_x=starting_point+new_momentum
            self.obj.save_moment(new_momentum)

        elif self.dim==2: 
         #   mean_1=np.average(self.model.sample(starting_point+0.9*self.obj.return_moment(),size=1).flatten())
        #    mean_2=np.average(self.model_1.sample(starting_point+0.9*self.obj.return_moment(),size=1).flatten())
            mean_1=np.average(self.model.sample(starting_point,size=2).flatten())
            mean_2=np.average(self.model_1.sample(starting_point,size=2).flatten())
         #   mean_1, var_1 = self.model.predict(starting_point)
         #   mean_2, var_2 = self.model_1.predict(starting_point)
            mean_1,mean_2=self.obj.perform_transform(mean_1,mean_2)
            print(mean_1,mean_2)
            mean=np.append(mean_1.item(), mean_2.item())
      #      m,v=self.obj.return_m_v()
      #      m_new=0.9*m+(1-0.9)*mean
      #      v_new=0.999*v+(1-0.999)*np.square(mean)
      #      self.obj.save_m_v(m_new,v_new)
      #      m_hat=m_new/(1-0.9**(self.count+1))
      #      v_hat=v_new/(1-0.999**(self.count+1))
      #      print(LR*((m_hat/(np.sqrt(v_hat)+1e-8))))
      #      new_x=starting_point+LR*((m_hat/(np.sqrt(v_hat)+1e-8)))

        #    new_momentum=0.9*self.obj.return_moment()+LR*mean
            new_x=starting_point+LR*mean
         #   self.obj.save_moment(new_momentum)
        return np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])


    def find_initial_point(self):
        x_tries = np.random.uniform(self.bounds[:, 0],self.bounds[:, 1],size=(1000, self.bounds.shape[0]))
        ys = self.min_var(x_tries)
        x_min = x_tries[np.random.choice(np.where(ys == ys.min())[0])]
        min_acq = ys.min()
        # Explore the parameter space more throughly
        x_seeds = np.random.uniform(self.bounds[:, 0],self.bounds[:, 1],size=(10, self.bounds.shape[0]))
        for x_try in x_seeds:
            res = minimize(lambda x: self.min_var(x.reshape(1, -1)),x_try.reshape(1, -1),bounds=self.bounds,method="L-BFGS-B")
            if not res.success:
                continue
            if min_acq is None or res.fun[0] <= min_acq:
                x_min = res.x
                min_acq = res.fun[0]
        return np.clip(x_min, self.bounds[:, 0],self.bounds[:, 1])

    def min_var(self,X):
        mean, var = self.model.predict(X)
        if self.dim==2:
            mean, var_1 = self.model.predict(X)
            mean, var_2 = self.model_1.predict(X)
            var=var_1+var_2
        std=np.sqrt(var)
        return std

        
    
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
    

