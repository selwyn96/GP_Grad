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

    def __init__(self, acq_name,bounds,model_gp,model,obj,Y,Y_S,y_max,X,Count,improv_counter):

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
            self.model_gp=model_gp
            self.y_max=y_max

    
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
    
    # randomly samples a point in the domain
    def _random(self):
        x= np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],size=( 1, self.bounds.shape[0]))[0]
        print(x)
        return x

    # GD algo with 5 local searches, same as GD with only m=5
    def _GD_5(self):
        m=5 # number of local searches
    #    LR=2
        max_search=1.07 # Maximum jump size
        # LR*unit vector= candidate points from which to search
        LR=np.sort(np.random.uniform(0, max_search,size=50)) # For local point selection
        LR2=np.sort(np.random.uniform(0, max_search,size=10)) # For Global selection 
       
       # Initialisation of the GD algo
        if(self.count==0):
            starting_point=self.X[-m:]  # chose the last m points initially to start
            reset_counter=np.zeros(m)   # reset_counter is used for local resets
            max_value=self.Y[-m:]       # max values in each local search
            cons_index=np.arange(len(self.Y)-m,len(self.Y))
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
                starting_point[index]=new_val
                reset_counter[index]=1
                self.obj.save_value(starting_point,reset_counter,last_UCB,max_value,index,X_TS,Y_TS,cons_index)
                return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])
                
            
            # Local reset (counter check + check if all gradients are above a min value)
       #     if(reset_counter[last_m]==1 or all(x<(0.01* max_search) for x in np.abs(self.obj.return_grad()))):


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
                a = (mean -self.y_max-0.1)
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
            new_x=np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])
            mean, var = self.model_gp.predict(new_x)
            std=np.sqrt(var)
         #   a = (mean - self.y_max)
        #    z = a / std
        #    improve= a * norm.cdf(z) + std * norm.pdf(z)
        #    z = (mean - self.y_max-0.01)/(std)
         #   improve= norm.cdf(z)
            improve=mean
            index=np.random.choice(np.where(improve == improve.max())[0])
            EI=np.append(EI,improve[index])
            index_values=np.append(index_values,index)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        index_i=index_values[index_j]
        return_value=starting_point[val_to_continue]+LR[index_j]*mean_test[int(index_i)]
        starting_point[val_to_continue]=np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])
    #    sampled_values=self.model_gp.sample(np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1]),size=20).flatten()
    #    count=np.count_nonzero(sampled_values > (1.1*max_value[val_to_continue]))/20
        # Local restart
        mean_original=np.std(self.Y)*EI[index_j]+np.mean(self.Y)
        if(mean_original<1.01*max_value[val_to_continue]):
            X_TS=np.vstack((X_TS, self.X[cons_index[val_to_continue]]))
            Y_TS=np.append(Y_TS,self.Y_S[cons_index[val_to_continue]])
            new_val=self._TS_GD(X_TS,Y_TS)
            starting_point[val_to_continue]=new_val
            reset_counter[val_to_continue]=1
            self.obj.save_grad(LR[index_j]*mean_test[int(index_i)])
            self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue,X_TS,Y_TS,cons_index)
            return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])

        self.obj.save_grad(LR[index_j]*mean_test[int(index_i)])
        self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue,X_TS,Y_TS,cons_index)
        
        return np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])
    
    # GD algo with 1 local searches
    def _GD(self):
        m=1# number of local searches
    #    LR=2
        max_search=1.07 # Maximum jump size
        # LR*unit vector= candidate points from which to search
        LR=np.sort(np.random.uniform(0, max_search,size=50)) # For local point selection
        LR2=np.sort(np.random.uniform(0, max_search,size=10)) # For Global selection 
       
       # Initialisation of the GD algo
        if(self.count==0):
            starting_point=self.X[-m:]  # chose the last m points initially to start
            reset_counter=np.zeros(m)   # reset_counter is used for local resets
            max_value=self.Y[-m:]       # max values in each local search
            cons_index=np.arange(len(self.Y)-m,len(self.Y))
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
                starting_point[index]=new_val
                reset_counter[index]=1
                self.obj.save_value(starting_point,reset_counter,last_UCB,max_value,index,X_TS,Y_TS,cons_index)
                return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])
                
            
            # Local reset (counter check + check if all gradients are above a min value)
       #     if(reset_counter[last_m]==1 or all(x<(0.01* max_search) for x in np.abs(self.obj.return_grad()))):


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
                a = (mean -self.y_max-0.1)
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
            new_x=np.clip(new_x, self.bounds[:, 0],self.bounds[:, 1])
            mean, var = self.model_gp.predict(new_x)
            std=np.sqrt(var)
         #   a = (mean - self.y_max)
        #    z = a / std
        #    improve= a * norm.cdf(z) + std * norm.pdf(z)
        #    z = (mean - self.y_max-0.01)/(std)
         #   improve= norm.cdf(z)
            improve=mean
            index=np.random.choice(np.where(improve == improve.max())[0])
            EI=np.append(EI,improve[index])
            index_values=np.append(index_values,index)
        index_j=np.random.choice(np.where(EI == EI.max())[0])
        index_i=index_values[index_j]
        return_value=starting_point[val_to_continue]+LR[index_j]*mean_test[int(index_i)]
        starting_point[val_to_continue]=np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1])
    #    sampled_values=self.model_gp.sample(np.clip(return_value, self.bounds[:, 0],self.bounds[:, 1]),size=20).flatten()
    #    count=np.count_nonzero(sampled_values > (1.1*max_value[val_to_continue]))/20
        # Local restart
        mean_original=np.std(self.Y)*EI[index_j]+np.mean(self.Y)
        if(mean_original<1.01*max_value[val_to_continue]):
            X_TS=np.vstack((X_TS, self.X[cons_index[val_to_continue]]))
            Y_TS=np.append(Y_TS,self.Y_S[cons_index[val_to_continue]])
            new_val=self._TS_GD(X_TS,Y_TS)
            starting_point[val_to_continue]=new_val
            reset_counter[val_to_continue]=1
            self.obj.save_grad(LR[index_j]*mean_test[int(index_i)])
            self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue,X_TS,Y_TS,cons_index)
            return np.clip(new_val, self.bounds[:, 0],self.bounds[:, 1])

        self.obj.save_grad(LR[index_j]*mean_test[int(index_i)])
        self.obj.save_value(starting_point,reset_counter,UCB,max_value,val_to_continue,X_TS,Y_TS,cons_index)
        
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
        ls,var=self.model_gp.Hyper()
        gp.set_hyper(ls,var)

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
    
    def _Restart(self):
        coordinates= self.sobol.draw(5000).cpu().numpy() * (self.ub - self.lb) + self.lb
        X_tries=coordinates
     #   samples = self.model_gp.sample(X_tries,size=1)
        mean, var = self.model_gp.predict(X_tries)
        samples= mean+np.log(3+self.count)*np.sqrt(var)
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
    

