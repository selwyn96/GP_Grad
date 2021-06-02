import sys
import torch
from sklearn.preprocessing import MinMaxScaler
from methods import methods
import numpy as np
from  utils import optimise_acq_func,acq_maximize,plot_posterior,plot_posterior_grad,plot_posterior_1d,plot_posterior_grad_1d,Momentum
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
     def __init__(self, func,bounds,Noise,Noise_level, acq_name,device='cuda',verbose=1):
          """
          We use the X and Y in original scale for gp_grad since we need to compare plots for the test situation. 
          we use scaled values of X and Y for creating the gp over observations converges faster to an optimum
          When we run a acq function using Gradient descent we could possible use the scaled values
           """
          self.X = None # The sampled point in original domain
          self.Y = None  # original output
          self.X_S=None   # scaled output (The input  is scaled [0,1] in all D) 
          self.Y_S=None   # scaled inpout ( output is scaled as  (Y - mu) / sigma )
          self.obj=None
          self.bounds=bounds # original bounds
          self.dim = len(bounds)
          self.bounds_s=np.array([np.zeros(self.dim), np.ones(self.dim)]).T  # scaled bounds
          self.func = func
          self.acq_name = acq_name
          scaler = MinMaxScaler()  # Tranform from orignal to scaled vales
          scaler.fit(self.bounds.T)
          self.Xscaler=scaler
          self.verbose=verbose
       #   self.gp=GaussianProcess(self.bounds_s,verbose=verbose) # GP over observed values
          self.time_opt=0
          self.count=0 # keeps a count of no of times a new value is sampled
          self.var=1 
          self.ls=1
          self.Noise=Noise
          self.Noise_level=Noise_level
          self.Noise_S=Noise_level
          

      
     
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
          self.Noise_S=self.Noise_level/np.std(self.Y)

     def find_kernel(self):
          gp_test=GaussianProcess(self.bounds,self.Noise, self.Noise_level,verbose=self.verbose) 
          X_test= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 10000, self.bounds.shape[0]))
          Y_try=self.func(X_test)
          gp_test.fit(X_test, Y_try)
          gp_test.optimise()
          var,ls=gp_test.Hyper()
          return(var,ls) 
              
     def sample_new_value(self):
     
          self.gp=GaussianProcess(self.bounds_s,self.Noise, self.Noise_level,verbose=self.verbose) # Acq function uses this GP to estimate next point to smaple 
          ur = unique_rows(self.X_S)
          self.gp.fit(self.X_S[ur], self.Y_S[ur])
        #  if  len(self.Y)%(3)==0:
       #        self.gp.optimise()
      #    if self.count==0:
       #        self.var,self.ls= self.find_kernel() # If we need to find the approx kernel of the function
          
          gp_grad_0=GaussianProcess_grad(self.bounds_s,self.Noise, self.Noise_S,D=0,verbose=self.verbose) # Create a GP for derivative in D=0
          gp_grad_1=GaussianProcess_grad(self.bounds_s,self.Noise, self.Noise_S ,D=1,verbose=self.verbose) # Create a GP for derivative in D=1
          gp_grad_0.set_hyper(self.ls,self.var) # setting the hyperparameters
          gp_grad_1.set_hyper(self.ls,self.var)
          gp_grad_0.fit(self.X_S[ur], self.Y_S[ur]) # fitting to the data which has been sampled
          gp_grad_1.fit(self.X_S[ur], self.Y_S[ur])
          plot_posterior_grad(self.bounds,gp_grad_0,gp_grad_1,self.X,self.Y,self.Noise,self.count) # Creating the Plot 

       #   gp_test= GaussianProcess(self.bounds,self.Noise, self.Noise_level,verbose=self.verbose)
       #   gp_test.fit(self.X, self.Y)
       #   plot_posterior_1d(self.bounds,gp_test,self.X, self.Y,self.Noise,self.count)
       #   plot_posterior_grad_1d(self.bounds,gp_grad_0,self.X, self.Y,self.Noise,self.count)
          
          if(self.count==0):  # creating object that saves the momentum values
               self.obj=Momentum()  

          y_max=max(self.Y_S)
          no_val_samp=len(self.Y_S) # For gpucb Beta 
          start_opt=time.time()
          # X_val is the new point that is sampled 
          if(self.acq_name=='random' or self.acq_name=='TS' or self.acq_name=='MES' or self.acq_name=='GD'):
               objects =methods(self.acq_name,self.bounds_s,gp_grad_0,gp_grad_1,self.obj,self.Y,self.X_S[len(self.X_S)-1],self.count)
               x_val=objects.method_val()
          else:
               x_val= optimise_acq_func(model=self.gp,bounds=self.bounds_s,y_max=y_max,sample_count=no_val_samp,acq_name=self.acq_name)

          # record the optimization time
          finished_opt=time.time()
          elapse_opt=finished_opt-start_opt
          self.time_opt=np.hstack((self.time_opt,elapse_opt))
          
           # Saving new values of X, Y 
          x_val_ori=self.Xscaler.inverse_transform(np.reshape(x_val,(-1,self.dim)))
        #  x_val_ori=x_val
        
          y_actual= self.func(x_val_ori) 
          self.X_S = np.vstack((self.X_S, x_val.reshape((1, -1))))
          self.X=np.vstack((self.X, x_val_ori))
          self.Y = np.append(self.Y, self.func(x_val_ori))
          self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)
          self.Noise_S=self.Noise_level/np.std(self.Y)
          self.count=self.count+1

          return x_val,y_actual

# Function to set Hypers
     def set_ls(self,lengthscale,variance):
        self.gp.set_hyper(lengthscale,variance)
     
     
     
           
        
     