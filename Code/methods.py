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


class methods(object):
    # Contain additional acq functions (slightly more complex ones)

    def __init__(self, acq_name,bounds,model_gp,model,obj,Y,X,Count,improv_counter):

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
            self.X=X
            self.count=Count
            self.improv_counter=improv_counter
            self.obj=obj
            self.model_gp=model_gp

    
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

    
    def _GD(self):
        m=5 # number of local searches
    #    LR=2
        max_search=0.15 # Maximum jump size
        # LR*unit vector= candidate points from which to search
        LR=np.sort(np.random.uniform(0, max_search,size=50)) # For local point selection
        LR2=np.sort(np.random.uniform(0, max_search,size=10)) # For Global selection 
       
       # Initialisation of the GD algo
        if(self.count==0):
            starting_point=self.X[-m:]  # chose the last m points initially to start
            reset_counter=np.zeros(m)   # reset_counter is used for local resets
            max_value=self.Y[-m:]       # max values in each local search
        else:    
            starting_point,reset_counter,last_UCB,max_value,last_m=self.obj.return_value() # Keeping a track of values from previous iterations

            # update counter if the sampled value isnt greater than local max by 2%
            if(self.Y[len(self.Y)-1]<(max_value[last_m]+0.02*max_value[last_m])):
                reset_counter[last_m]=reset_counter[last_m]+1
            
            # update new local max
            if(self.Y[len(self.Y)-1]>max_value[last_m]):
                max_value[last_m]=self.Y[len(self.Y)-1]

            # Global reset (search with lowest UCB from the last iteration is reset)
            if(self.improv_counter>=5):
                new_val=self._TS()
                index=np.random.choice(np.where(last_UCB == last_UCB.min())[0])
                starting_point[index]=new_val
                max_value[index]=float('-inf')
                reset_counter[index]=0
            
            # Local reset (counter check + check if all gradients are above a min value)
            if(reset_counter[last_m]>=5 or all(x<0.0001 for x in np.abs(self.obj.return_grad()))):
                new_val=self._TS()
                starting_point[last_m]=new_val
                max_value[last_m]=float('-inf')
                reset_counter[last_m]=0
               
        # Global search
        UCB=[]
        for i in range(0,m):
            mean_temp=np.array([])
            for l in range(0,10): # 10 candidate unit vectors for each m are compared
                tempo=np.array([])
                for ob in self.model: # ob is a list containing all the grad_gp objects for different D values
                    temp_val=np.average(ob.sample(starting_point[i],size=5)) 
                    tempo=np.append(tempo,temp_val.flatten())
                l2=np.sum(np.power((tempo),2))**(0.5)
                tempo=tempo/l2 # unit vector 
                if(l==0):
                    mean_temp=tempo
                else:
                    mean_temp=np.vstack((mean_temp,tempo))
            mean= mean_temp
            

            #    mean=np.tanh(mean)
            UCB_temp=[]
            # At a global level EI/UCB is values are used to select which GD search to continue
            for j in range(0,10):
                mean_used=LR2[j]*mean
                new_x=starting_point[i]+mean_used
                mean, var = self.model_gp.predict(new_x)
           #     beta=np.log(self.count+1)
           #     improve= mean + np.sqrt(beta) * np.sqrt(var)
                std=np.sqrt(var)
                a = (mean - np.max(self.Y)-0.1)
                z = a / std
                improve= a * norm.cdf(z) + std * norm.pdf(z)
                index=np.random.choice(np.where(improve == improve.max())[0])
                UCB_temp=np.append(UCB_temp,improve[index])
            
            index=np.random.choice(np.where(UCB_temp ==  UCB_temp.max())[0])
            UCB=np.append(UCB,UCB_temp[index])
        val_to_continue=np.random.choice(np.where(UCB == UCB.max())[0])
        
        # Local search
        mean_test=[]
        for i in range(0,10):
            mean=np.array([])
            for ob in self.model:
                mean_temp=np.average(ob.sample(starting_point[val_to_continue],size=5))
                mean=np.append(mean,mean_temp.flatten())
            #    mean=np.append(mean,ob.sample(starting_point[val_to_continue],size=1).flatten().item())
                 
            l2=np.sum(np.power((mean),2))**(0.5)
            mean=mean/l2
         #   mean=np.tanh(mean)
            if(i==0):
                mean_test=mean
            else:
                mean_test=np.vstack((mean_test,mean))
        index_values=[]
        EI=[]
        # At the local level mean is used to select which point to sample in the local search
        for j in range(0,50):
            mean_used=LR[j]*mean_test
            new_x=starting_point[val_to_continue]+mean_used
            mean, var = self.model_gp.predict(new_x)
            std=np.sqrt(var)
         #   a = (mean - np.max(self.Y))
        #    z = a / std
        #    improve= a * norm.cdf(z) + std * norm.pdf(z)
        #    z = (mean - np.max(self.Y)-0.01)/(std)
         #   improve= norm.cdf(z)
            improve=mean
            index=np.random.choice(np.where(improve == improve.max())[0])
            EI=np.append(EI,improve[index])
            index_values=np.append(index_values,index)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        index_i=index_values[index_j]
        return_value=starting_point[val_to_continue]+LR[index_j]*mean_test[int(index_i)]
        starting_point[val_to_continue]=np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])
        self.obj.save_grad(LR[index_j]*mean_test[int(index_i)])
        self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue)
        
        return np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])


        
    
    # Gradient descent algo

    # def _GD(self):
    #     m=5 # number of local searches
    #     LR=0.1
    # #    LR=0.1+0.5*(0.5-0.1)*(1+math.cos((self.count/20)*math.pi))
    #     if(self.count==0):
    #         starting_point=self.find_initial_point()
    #      #   starting_points=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( m, self.bounds.shape[0]))
    #     else:
    #             starting_point=self.present_val
    #             if(all(x<0.1 for x in np.abs(self.obj.return_grad())) or self.improv_counter>=10):
    #                 starting_point=self._TS()
                   
                    

    #              #   starting_point=self._TS()

    #     if self.dim==1:
    #      #   mean, var = self.model.predict(starting_point+0.9*self.obj.return_moment())
    #         mean = self.model.sample(starting_point+0.9*self.obj.return_moment(),size=1).flatten()
    #         new_momentum=0.9*self.obj.return_moment()+LR*mean
    #         new_x=starting_point+new_momentum
    #         self.obj.save_moment(new_momentum)

    #     elif self.dim==2: 
    #     #   mean_1, var_1 = self.model.predict(starting_point)
    #      #   mean_2, var_2 = self.model_1.predict(starting_point)
    #         mean_test=[]
    #         for i in range(0,20):
    #             mean_1=np.average(self.model.sample(starting_point,size=1).flatten())
    #             mean_2=np.average(self.model_1.sample(starting_point,size=1).flatten())
    #          #   mean_1,mean_2=self.obj.perform_transform(mean_1,mean_2)
    #          #   self.obj.get_max_min(max_1,min_1,max_2,min_2)
    #             mean=np.append(mean_1.item(), mean_2.item())
    #             mean=np.tanh(mean)
    #             if(i==0):
    #                 mean_test=mean
    #             else:
    #                 mean_test=np.vstack((mean_test,mean))
    #         new_x=starting_point+LR*mean_test
    #         mean, var = self.model_gp.predict(new_x)
    #         std=np.sqrt(var)
    #         a = (mean - np.max(self.Y))
    #         z = a / std
    #         improve= a * norm.cdf(z) + std * norm.pdf(z)
    #      #   z = (mean - np.max(self.Y))/(std)
    #      #   improve= norm.cdf(z)
    #         index=np.random.choice(np.where(improve == improve.max())[0])
    #         self.obj.save_grad(mean_test[index])
    #       #  mean_1,mean_2=self.obj.perform_transform(mean_test[index][0],mean_test[index][1])
        
    #     return np.clip(new_x[index], self.bounds[:, 0],self.bounds[:, 1])

        




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
        coordinates= self.sobol.draw(2000).cpu().numpy() * (self.ub - self.lb) + self.lb
        X_tries=coordinates
        samples = self.model_gp.sample(X_tries,size=1)
        index=np.random.choice(np.where(samples == samples.max())[0])
        return(X_tries[index])
    
    def _Restart(self):
        coordinates= self.sobol.draw(5000).cpu().numpy() * (self.ub - self.lb) + self.lb
        X_tries=coordinates
     #   samples = self.model_gp.sample(X_tries,size=1)
        mean, var = self.model_gp.predict(X_tries)
        samples=mean + np.sqrt(np.log(self.count+3)) * np.sqrt(var)
        index=np.random.choice(np.where(samples == samples.max())[0])
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
    

