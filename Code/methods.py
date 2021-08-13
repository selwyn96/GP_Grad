import sys
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
from scipy.stats import norm
from numpy import linalg as LA


class methods(object):
    # Contain additional acq functions (slightly more complex ones)

    def __init__(self, acq_name,bounds,model_gp,model,obj,turbo,Y,Y_S,y_max,X,Count,improv_counter):

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
            self.Y_S=Y_S
            self.X=X
            self.count=Count
            self.improv_counter=improv_counter
            self.obj=obj
            self.turbo=turbo
            self.model_gp=model_gp
            self.y_max=y_max
            self.succtol = 1
            self.failtol=5

    
    def method_val(self):
  
        if self.acq_name == 'random' :
            return self._random()
        
        if self.acq_name == 'TS' :
            return self._TS()

        if self.acq_name == 'MES' :
            return self._MES()

        if self.acq_name == 'GD' :
            return self._GD()

        if self.acq_name == 'GD_5' :
            return self._GD_5()

        if self.acq_name == 'GD_Turbo' :
            return self._GD_Turbo()
    
    
    # randomly samples a point in the domain
    def _random(self):
        x= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 1, self.bounds.shape[0]))[0]
        return x

    # GD algo with 5 local searches, same as GD with only m=5
    def _GD_5(self):
        m=5 # number of local searches
    #    LR=2
        max_search=5 # Maximum jump size
        # LR*unit vector= candidate points from which to search
        LR=np.sort(np.random.uniform(0, max_search,size=500)) # For local point selection
        LR2=np.sort(np.random.uniform(0, max_search,size=10)) # For Global selection 
       
       # Initialisation of the GD algo
        if(self.count==0):
            index_max=np.argsort(self.Y)[::-1][:m]
            starting_point=self.X[index_max] # chose the last m points initially to start
            reset_counter=np.zeros(m)   # reset_counter is used for local resets
            max_value=np.atleast_1d(self.Y[index_max])    # max values in each local search
            cons_index=np.atleast_1d(index_max)
            X_TS=self.X
            Y_TS=self.Y_S
        else:    
            starting_point,reset_counter,last_UCB,max_value,last_m,X_TS,Y_TS,cons_index=self.obj.return_value() # Keeping a track of values from previous iterations
        
            # update counter if the sampled value isnt greater than local max by 2%
        #    if(self.Y[len(self.Y)-1]<(max_value[last_m]+0.001*max_value[last_m])):
        #        reset_counter[last_m]=reset_counter[last_m]+1
            
            # update new local max
            if(reset_counter[last_m]==1):
                 max_value[last_m]=self.Y[len(self.Y)-1]
                 reset_counter[last_m]=0
                 cons_index[last_m]=len(self.Y)-1

            elif(self.Y[len(self.Y)-1]>max_value[last_m]):
                max_value[last_m]=self.Y[len(self.Y)-1]
                cons_index[last_m]=len(self.Y)-1
            
            # Global reset (search with lowest UCB from the last iteration is reset)
            if(self.improv_counter>=10):
                index=np.random.choice(np.where(last_UCB == last_UCB.min())[0])
                X_TS=np.vstack((X_TS, self.X[cons_index[index]]))
                Y_TS=np.append(Y_TS,self.Y_S[cons_index[index]])
                new_val=self._TS_GD(X_TS,Y_TS)
              #  new_val=self._random()
                starting_point[index]=new_val
                reset_counter[index]=1
                self.obj.save_value(starting_point,reset_counter,last_UCB,max_value,index,X_TS,Y_TS,cons_index)
                return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])
                
            
            # Local reset (counter check + check if all gradients are above a min value)
       #     if(reset_counter[last_m]==1 or all(x<(0.01* max_search) for x in np.abs(self.obj.return_grad()))):


        # Global search
        UCB=[]
        for i in range(0,m):
            tempo=np.array([])
            for ob in self.model: # ob is a list containing all the grad_gp objects for different D values
                temp_val=np.average(ob.sample(starting_point[i],size=5)) 
                tempo=np.append(tempo,temp_val.flatten())
         #   l2=np.sum(np.power((tempo),2))**(0.5)
            l2=LA.norm(tempo)
            tempo=tempo/l2 # unit vector 
            mean= tempo
            
            UCB_temp=[]
            # At a global level EI/UCB is values are used to select which GD search to continue
            for j in range(0,10): 
                mean_used=LR2[j]*mean
                new_x=starting_point[i]+mean_used
                mean, var = self.model_gp.predict(new_x)
           #     beta=np.log(self.count+1)
           #     improve= mean + np.sqrt(beta) * np.sqrt(var)
                std=np.sqrt(var)
                a = (mean -self.y_max-0.1)
                z = a / std
                improve= a * norm.cdf(z) + std * norm.pdf(z)
                UCB_temp=np.append(UCB_temp,improve)
            
            index=np.random.choice(np.where(UCB_temp ==  UCB_temp.max())[0])
            UCB=np.append(UCB,UCB_temp[index])
        val_to_continue=np.random.choice(np.where(UCB == UCB.max())[0])
        
        # Local search
        mean=np.array([])
        for ob in self.model:
            mean_temp=np.average(ob.sample(starting_point[val_to_continue],size=5))
            mean=np.append(mean,mean_temp.flatten())   
        l2=np.sum(np.power((mean),2))**(0.5)
        mean_test=mean/l2
            
        EI=[]
        # At the local level mean is used to select which point to sample in the local search
        for j in range(0,500):
            mean_used=LR[j]*mean_test
            new_x=starting_point[val_to_continue]+mean_used
            new_x=np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])
            mean=self.model_gp.sample(new_x[np.newaxis,:],size=1)
         #   mean, var = self.model_gp.predict(new_x)
            std=np.sqrt(var)
         #   a = (mean - self.y_max)
        #    z = a / std
        #    improve= a * norm.cdf(z) + std * norm.pdf(z)
        #    z = (mean - self.y_max-0.01)/(std)
         #   improve= norm.cdf(z)
            improve=mean
            EI=np.append(EI,improve)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        return_value=starting_point[val_to_continue]+LR[index_j]*mean_test
        starting_point[val_to_continue]=np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])
    #    sampled_values=self.model_gp.sample(np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1]),size=20).flatten()
    #    count=np.count_nonzero(sampled_values > (1.1*max_value[val_to_continue]))/20
        # Local restart
        mean_original=np.std(self.Y)*EI[index_j]+np.mean(self.Y)
        if(mean_original<1.01*max_value[val_to_continue]):
            X_TS=np.vstack((X_TS, self.X[cons_index[val_to_continue]]))
            Y_TS=np.append(Y_TS,self.Y_S[cons_index[val_to_continue]])
            new_val=self._TS_GD(X_TS,Y_TS)
          #  new_val=self._random()
            starting_point[val_to_continue]=new_val
            reset_counter[val_to_continue]=1
            self.obj.save_grad(LR[index_j]*mean_test)
            self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue,X_TS,Y_TS,cons_index)
            return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])

        self.obj.save_grad(LR[index_j]*mean_test)
        self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue,X_TS,Y_TS,cons_index)
        
        return np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])
    
    # GD algo with 1 local searches
    def _GD(self):
        m=1# number of local searches
         # Default
        
        if(self.count==0):
            index_max=np.argmax(self.Y)
            starting_point=self.X[index_max][np.newaxis,:]  # chose the largest (highest Y) point to start
            reset_counter=np.zeros(m)   # reset_counter is used for local resets
            max_value=np.atleast_1d(self.Y[index_max])      # max values in each local search
            cons_index=np.atleast_1d(index_max)
            X_TS=self.X
            Y_TS=self.Y_S
            succcount=0
            failcount=0
            val_to_continue=0
            self.max_search = 15
            
        else:    
            previos_value,starting_point,reset_counter,max_value,last_m,X_TS,Y_TS,cons_index,succcount,failcount,self.max_search=self.turbo.return_value() # Keeping a track of values from previous iterations
            val_to_continue=last_m

            if(reset_counter[last_m]==0):
                if (self.Y[len(self.Y)-1] > max_value[last_m] + 1e-3 * math.fabs(max_value[last_m])):
                    succcount += 1
                    failcount = 0
                else:
                    succcount = 0
                    failcount += 1
                # if succcount >= self.succtol: 
                #     self.max_search = min([2 * self.max_search,10])
                #     succcount = 0
                # elif failcount >= self.failtol:  
                #     self.max_search /= 2
                #     failcount = 0

            # update new local max
            if(reset_counter[last_m]==1):
                 max_value[last_m]=self.Y[len(self.Y)-1]
                 reset_counter[last_m]=0
                 cons_index[last_m]=len(self.Y)-1
                 failcount = 0
                 succcount=0

            elif(self.Y[len(self.Y)-1]>max_value[last_m]):
                max_value[last_m]=self.Y[len(self.Y)-1]
                cons_index[last_m]=len(self.Y)-1
                print(f"{self.count}) Grad Step: {max_value[last_m]:.4}" )
                sys.stdout.flush()

            elif(self.Y[len(self.Y)-1]<max_value[last_m]):
                starting_point[last_m]=previos_value[last_m]

            # Restart
            # self.max_search <=  6*0.8 ** 3
            if(failcount>self.failtol): # 2 is original max_search
                X_TS=np.vstack((X_TS, self.X[cons_index[last_m]]))
                Y_TS=np.append(Y_TS,self.Y_S[cons_index[last_m]])
                new_val=self._random()
            #    new_val=self._TS_GD(self.X,self.Y_S)
            #    new_val=self._Restart(starting_point[last_m])
            #    new_val=self._TS()
                print('Restart')
                starting_point[last_m]=new_val
                reset_counter[last_m]=1
                self.max_search = 15
                self.turbo.save_value(starting_point,starting_point,reset_counter,max_value,last_m,X_TS,Y_TS,cons_index,succcount,failcount,self.max_search)
                return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])
  
        LR=np.sort(np.random.uniform(0, self.max_search,size=1000)) # For local point selection
        
        mean=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 1, self.bounds.shape[0]))[0]
        l2=np.sum(np.power((mean),2))**(0.5)
        mean_test=mean/l2

        EI=[]
        # At the local level mean is used to select which point to sample in the local search
        for j in range(0,1000):
            mean_used=LR[j]*mean_test
            new_x=starting_point[val_to_continue]+mean_used
            new_x=np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])
            mean=self.model_gp.sample(new_x[np.newaxis,:],size=1)
      #      mean, var = self.model_gp.predict(new_x)
            improve=mean
            EI=np.append(EI,improve)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        print(LR[index_j])
        return_value=starting_point[val_to_continue]+LR[index_j]*mean_test
        previos_value=starting_point

        self.turbo.save_value(previos_value,np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])[np.newaxis,:],reset_counter,max_value,val_to_continue,X_TS,Y_TS,cons_index,succcount,failcount,self.max_search)
    
        return np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])


    def _GD_Turbo(self):
        m=1# number of local searches
         # Default
        
        if(self.count==0):
            index_max=np.argmax(self.Y)
            starting_point=self.X[index_max][np.newaxis,:]  # chose the largest (highest Y) point to start
            reset_counter=np.zeros(m)   # reset_counter is used for local resets
            max_value=np.atleast_1d(self.Y[index_max])      # max values in each local search
            cons_index=np.atleast_1d(index_max)
            X_TS=self.X
            Y_TS=self.Y_S
            succcount=0
            failcount=0
            val_to_continue=0
            self.max_search = 5
            
        else:    
            previos_value,starting_point,reset_counter,max_value,last_m,X_TS,Y_TS,cons_index,succcount,failcount,self.max_search=self.turbo.return_value() # Keeping a track of values from previous iterations
            val_to_continue=last_m
            print(previos_value)
            print(starting_point)

            if(reset_counter[last_m]==0):
                if (self.Y[len(self.Y)-1] > max_value[last_m] + 1e-3 * math.fabs(max_value[last_m])):
                    succcount += 1
                    failcount = 0
                else:
                    succcount = 0
                    failcount += 1
                if succcount >= self.succtol:
                    self.max_search=6
              #      self.max_search = min([2 * self.max_search,10])
                    succcount = 0
                elif failcount >= self.failtol:
                    self.max_search= max([6,6])
                 #   self.max_search /= 2
                 #   failcount = 0

            # update new local max
            if(reset_counter[last_m]==1):
                 max_value[last_m]=self.Y[len(self.Y)-1]
                 reset_counter[last_m]=0
                 cons_index[last_m]=len(self.Y)-1
                 failcount = 0
                 succcount=0

            elif(self.Y[len(self.Y)-1]>max_value[last_m]):
                max_value[last_m]=self.Y[len(self.Y)-1]
                cons_index[last_m]=len(self.Y)-1
                print(f"{self.count}) Grad Step: {max_value[last_m]:.4}" )
                sys.stdout.flush()

            elif(self.Y[len(self.Y)-1]<max_value[last_m]):
                starting_point[last_m]=previos_value[last_m]

            # Restart
            # self.max_search <=  6*0.8 ** 3
            if(failcount>self.failtol): # 2 is original max_search
                X_TS=np.vstack((X_TS, self.X[cons_index[last_m]]))
                Y_TS=np.append(Y_TS,self.Y_S[cons_index[last_m]])
                new_val=self._random()
            #    new_val=self._TS_GD(self.X,self.Y_S)
            #    new_val=self._Restart(starting_point[last_m])
            #    new_val=self._TS()
                print('Restart')
                starting_point[last_m]=new_val
                reset_counter[last_m]=1
                self.max_search = 5
                self.turbo.save_value(starting_point,starting_point,reset_counter,max_value,last_m,X_TS,Y_TS,cons_index,succcount,failcount,self.max_search)
                return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])

        print(self.max_search)
        LR=np.sort(np.random.uniform(0, self.max_search,size=1000)) # For local point selection
    

        mean=np.array([])
        for ob in self.model:
            mean_temp=np.average(ob.sample(starting_point[val_to_continue],size=5))
            mean=np.append(mean,mean_temp.flatten())    
            
        l2=np.sum(np.power((mean),2))**(0.5)
        mean_test=mean/l2

        EI=[]
        # At the local level mean is used to select which point to sample in the local search
        for j in range(0,1000):
            mean_used=LR[j]*mean_test
            new_x=starting_point[val_to_continue]+mean_used
            new_x=np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])
            mean=self.model_gp.sample(new_x[np.newaxis,:],size=1)
      #      mean, var = self.model_gp.predict(new_x)
            improve=mean
            EI=np.append(EI,improve)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        print(LR[index_j])
        return_value=starting_point[val_to_continue]+LR[index_j]*mean_test
        previos_value=starting_point[val_to_continue]
     #   starting_point[val_to_continue]=np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])

        self.turbo.save_value(previos_value,np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])[np.newaxis,:],reset_counter,max_value,val_to_continue,X_TS,Y_TS,cons_index,succcount,failcount,self.max_search)
    
        return np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])






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
    def _TS_GD(self,X,Y):
        gp=GaussianProcess(self.bounds) # Acq function uses this GP to estimate next point to smaple 
        ur = unique_rows(X)
        gp.fit(X[ur],Y[ur])
        gp.optimise()
     #   ls,var=self.model_gp.Hyper()
      #  gp.set_hyper(ls,var)

        coordinates= self.sobol.draw(5000).cpu().numpy() * (self.ub - self.lb) + self.lb
        X_tries=coordinates
        samples = gp.sample(X_tries,size=1)
        index=np.random.choice(np.where(samples == samples.max())[0])
        return(X_tries[index])
    
    def _TS(self):
        coordinates= self.sobol.draw(5000).cpu().numpy() * (self.ub - self.lb) + self.lb
        X_tries=coordinates
        samples = self.model_gp.sample(X_tries,size=1)
        index=np.random.choice(np.where(samples == samples.max())[0])
        return(X_tries[index])
    
    def _Restart(self,current_value):
        X= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 1, self.bounds.shape[0]))[0]
        l2=np.sum(np.power((X),2))**(0.5)
        X=X/l2
        LR=np.sort(np.random.uniform(0, self.max_search,size=500))
        EI=np.array([])
        for j in range(0,500):
            mean=LR[j]*X
            new_x=current_value+mean
            new_x=np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])
            mean=self.model_gp.sample(new_x[np.newaxis,:],size=1)
            improve=mean
            EI=np.append(EI,improve)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        return_value=current_value+LR[index_j]*X
        return(return_value)

    
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
    

