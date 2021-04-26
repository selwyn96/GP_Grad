import sys
import torch
from sklearn.preprocessing import MinMaxScaler
from methods import methods
import numpy as np
from  utils import optimise_acq_func,acq_maximize,plot_posterior,plot_posterior_grad
from Gaussian import GaussianProcess
from Gaussian_grad import GaussianProcess_grad
from methods import methods
import time


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking
    :param a: array to trim repeated rows from
    :return: mask of unique rows
    """
    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class GP_action:
     def __init__(self, func,bounds, acq_name,device='cuda',verbose=1):
          self.X = None # The sampled point in original domain
          self.Y = None  # original output
          self.X_S=None   # scaled output (The input  is scaled [0,1] in all D)
          self.Y_S=None   # scaled inpout ( output is scaled as  (Y - mu) / sigma )
          self.bounds=bounds
          self.dim = len(bounds) # original bounds
          self.bounds_s=np.array([np.zeros(self.dim), np.ones(self.dim)]).T  # scaled bounds
          self.func = func
          self.acq_name = acq_name
          scaler = MinMaxScaler()  # Tranform for moving from orignal to scaled vales
          scaler.fit(self.bounds.T)
          self.Xscaler=scaler
          self.verbose=verbose
          self.gp=GaussianProcess(self.bounds_s,verbose=verbose)
          self.time_opt=0
          self.count=0 #keeps a count of no of times the function is called
     
     def initiate(self, seed,n_random_draws=3):

          ''' This function samples the intial 2-3 points'''
          np.random.seed(seed)
          X_test= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 3, self.bounds.shape[0]))
          self.X = np.asarray(X_test[0])
          self.X_S = self.Xscaler.transform(X_test[0].reshape((1, -1)))
          y_init=self.func(X_test[0])
          self.Y = np.asarray(y_init)
          for i in range (1,n_random_draws):
               self.X=np.vstack((self.X, X_test[i]))
               x_s=self.Xscaler.transform(X_test[i].reshape((1, -1)))
               self.X_S = np.vstack((self.X_S, x_s))
               self.Y = np.append(self.Y, self.func(X_test[i]))
          self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)      

     

     def sample_new_value(self):
          self.gp=GaussianProcess(self.bounds_s,verbose=self.verbose)
          ur = unique_rows(self.X_S)
          self.gp.fit(self.X_S[ur], self.Y_S[ur])
        #  if  len(self.Y)%(3)==0:
       #        self.gp.optimise()
          variance,lenghtscale=self.gp.Hyper()
          gp_grad=GaussianProcess_grad(self.bounds,verbose=self.verbose)
          gp_grad.set_hyper(lenghtscale,variance)
          gp_grad.fit(self.X[ur], self.Y[ur])
          gp2=GaussianProcess(self.bounds,verbose=self.verbose)
          gp2.fit(self.X[ur], self.Y[ur])
       #   plot_posterior(self.bounds,gp2,self.X,self.Y,self.count) 
          plot_posterior_grad(self.bounds,gp_grad,self.count)

          y_max=max(self.Y_S)
          no_val_samp=len(self.Y_S) # For gpucb Beta 
          start_opt=time.time()
          if(self.acq_name=='random' or self.acq_name=='TS' or self.acq_name=='MES'):
               objects =methods(self.acq_name,self.bounds_s,self.gp,self.Y_S )
               x_val=objects.method_val()
          else:
               x_val= optimise_acq_func(model=self.gp,bounds=self.bounds_s,y_max=y_max,sample_count=no_val_samp,acq_name=self.acq_name)

          # record the optimization time
          finished_opt=time.time()
          elapse_opt=finished_opt-start_opt
          self.time_opt=np.hstack((self.time_opt,elapse_opt))
          
           # Saving new values of X, Y 
          x_val_ori=self.Xscaler.inverse_transform(np.reshape(x_val,(-1,self.dim)))
        
          y_actual= self.func(x_val_ori[0]) 
          self.X_S = np.vstack((self.X_S, x_val.reshape((1, -1))))
          self.X=np.vstack((self.X, x_val_ori))
          self.Y = np.append(self.Y, self.func(x_val_ori[0]))
          self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)
          self.count=self.count+1

          return x_val,y_actual
# Function to set Hypers
     def set_ls(self,lengthscale,variance):
        self.gp.set_hyper(lengthscale,variance)
     
     
     
           
        
     