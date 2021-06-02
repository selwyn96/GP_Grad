import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import scipy
#from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
#import matplotlib as mpl
import matplotlib.cm as cm
from scipy.linalg import block_diag
from numpy.linalg import inv
class GaussianProcess_grad(object):
    def __init__ (self,SearchSpace,Noise=False,noise_delta=1e-8,D=0,verbose=0):
        self.noise_upperbound=noise_delta
        self.mycov=self.cov_RBF 
        self.mycov_11=self.cov_11 # Covariance between two partial derivatives
        self.mycov_01=self.cov_01 # Covariance between and observation and partial derivatives
        self.SearchSpace=SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler
        self.verbose=verbose
        self.dim=SearchSpace.shape[0]
        self.D=D # This is the dimension along which the grad GP is calculated (0 to N-1)
        
        self.hyper={}
        self.hyper['var']=1 # standardise the data
        self.hyper['lengthscale']=1 #to be optimised
        if(Noise==True) :
            self.noise_delta=noise_delta**2
        else:
            self.noise_delta=1e-8
        return None
   
        
    def fit(self,X,Y,IsOptimize=0):
        """
        Fit a Gaussian Process model
        X: input 2d array [N*d]
        Y: output 2d array [N*1]
        """       
        #self.X= self.Xscaler.transform(X) #this is the normalised data [0-1] in each column
        self.X=X
        self.Y=Y 
        
        if IsOptimize:
            self.hyper['lengthscale']=self.optimise()[0]         # optimise GP hyperparameters
            self.hyper['var']=self.optimise()[1]
        self.KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta     
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x !")
      
        self.L=scipy.linalg.cholesky(self.KK_x_x,lower=True)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        
    def cov_RBF(self,x1, x2,hyper):        
        """
        Radial Basic function kernel (or SE kernel)
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        Euc_dist=euclidean_distances(x1,x2)
        return variance*np.exp(-np.square(Euc_dist)/2*np.square(lengthscale))

    def cov_11(self,x1, x2,hyper):        
        """
        Check the notes for the expression
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        Euc_dist=euclidean_distances(x1,x2)
        Euc_dist_2=np.array((x1[:,self.D][:, np.newaxis] - x2[:,self.D]))
        Euc_dist_2=np.reshape(Euc_dist_2,Euc_dist.shape)

        a=np.exp(-np.square(Euc_dist)/2*np.square(lengthscale))
        div= np.square(Euc_dist_2)/np.square(lengthscale)
        b=(variance/np.square(lengthscale))*(1-div)
        return a*b
    
    def cov_01(self,x1, x2,hyper):        
        """
        Check the notes for the expression
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        Euc_dist=euclidean_distances(x1,x2)
        Euc_dist_2=np.array((x1[:,self.D][:, np.newaxis] - x2[:,self.D]))
        Euc_dist_2=np.reshape(Euc_dist_2,Euc_dist.shape)
        
        a=np.exp(-np.square(Euc_dist)/2*np.square(lengthscale))
        b=(variance/np.square(lengthscale))*(Euc_dist_2)
        return a*b
    

    def log_llk(self,X,y,hyper_values):
        
        #print(hyper_values)
        hyper={}
        hyper['var']=hyper_values[1]
        hyper['lengthscale']=hyper_values[0]
        noise_delta=self.noise_delta

        KK_x_x=self.mycov(X,X,hyper)+np.eye(len(X))*noise_delta     
        if np.isnan(KK_x_x).any(): #NaN
            print("nan in KK_x_x !")   

        try:
            L=scipy.linalg.cholesky(KK_x_x,lower=True)
            alpha=np.linalg.solve(KK_x_x,y)

        except: # singular
            return -np.inf
        try:
            first_term=-0.5*np.dot(self.Y.T,alpha)
            W_logdet=np.sum(np.log(np.diag(L)))
            second_term=-W_logdet

        except: # singular
            return -np.inf

        logmarginal=first_term+second_term-0.5*len(y)*np.log(2*3.14)
        
        #print(hyper_values,logmarginal)
        return np.asscalar(logmarginal)
    
    def set_hyper (self,lengthscale,variance):
        self.hyper['lengthscale']=lengthscale
        self.hyper['var']=variance
        
    def predict(self,Xtest,isOriScale=False):
        """
        ----------
        Xtest: the testing points  [N*d]
        Returns
        -------
        pred mean, pred var
        """    
        
        if isOriScale:
            Xtest=self.Xscaler.transform(Xtest)
            
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
        KK_xTest_xTest=self.mycov_11(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_x=self.mycov_01(self.X,Xtest,self.hyper)
        temp=np.dot(KK_xTest_x.T,inv(self.KK_x_x))
        mean=np.dot(temp,self.Y)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_x)
        std=np.reshape(np.diag(var),(-1,1))
        return  np.reshape(mean,(-1,1)),std 

        

   
   # sampling a point from the posterior
    def sample(self,X,size):
        m, var = self.predict(X)
        v=self.covar(X)
        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T
        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]
    
    # Returns the covariance matrix
    def covar(self,Xtest):
        """
        Returns Covariance matrix
        """    
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
        KK_xTest_xTest=self.mycov_11(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_x=self.mycov_01(self.X,Xtest,self.hyper)
        temp=np.dot(KK_xTest_x.T,inv(self.KK_x_x))
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_x)
        return  var