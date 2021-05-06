import numpy as np
from numpy import *
import math
from numpy.matlib import *
import GPy
from scipy.stats import multivariate_normal



def synthetic_input():
    X_1 = np.asarray([[-2, -1.6],[-2, -2.2], [-1.2, -1.5], [-1.6, 0.6]])
    Y_1 = np.asarray([[0.6], [0.4], [0.3], [-0.4]])
    X_2 = np.asarray([[-0.7, -0.5], [-0.5, 0.3], [0.1, -0.3], [0.3, -1], [0.7, -0.6], [0.3, 0.1]])
    Y_2 = np.asarray([[-0.7], [0.7], [1], [-0.3], [0.1], [0.4]])
    X_3 = np.asarray([[2.1, -2], [1.6, 0.1]])
    Y_3 = np.asarray([[0.7], [-0.35]])
    X_4 = np.asarray([[1.7, 1.9], [0.5, 1.], [0.2, 1.3], [1.2, 1.4]])
    Y_4 = np.asarray([[0.9], [0.7], [0.5], [0.5]])
    X_5 = np.asarray([[-2.1, 1.8]])
    Y_5 = np.asarray([[-0.5]])
    X = np.vstack([X_1, X_2, X_3, X_4, X_5])
    Y = np.vstack([Y_1, Y_2, Y_3, Y_4, Y_5])
    return X,Y




class Kean:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-4, 4), 'y': (-4, 4)}
        self.name='Kean'
    def func(self,coord):
        if(coord.ndim==1):
            coord=coord[np.newaxis,:]
        X1=coord[:,0]
        X2=coord[:,1]
        out =   np.abs((np.cos(X1)**4 + np.cos(X2)**4 - 2 * (np.cos(X1)**2) * (np.cos(X2)**2))) / np.sqrt(1*X1**2 + 1.5*X2**2)
        out=np.squeeze(out)
        return out * 1.5

class Shubert:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-3, 1), 'y': (-3, 1)}
        self.name='Shubert'
    def func(self,coord):
        out_0 = 0
        out_1 = 0
        for i in range(5):
            out_0 += (i + 1) * np.cos((i + 2) * coord[0] + (i + 1))
        for i in range(5):
            out_1 += (i + 1) * np.sin((i + 2) * coord[1] + (i + 1))
        return (out_0 * out_1 + 220)/200


class Branin:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-5, 10), 'y': (0, 15)}
        self.name='Branin'
    def func(self,coord):
        PI = math.pi
        a = 1
        b = 5.1/(4*pow(PI, 2))
        c = 5/PI
        r = 6   
        s = 10
        t = 1/(8*PI)
        out = a*(coord[1] - b*coord[0]**2 + c*coord[0] -r)**2 + s*(1-t)*np.cos(coord[0]) + s
        return - out


class Hartmann_3:
    def __init__(self):
        self.input_dim=3
        self.bounds={'x': (0, 1), 'y': (0, 1),'z':(0,1)}
        self.name='Hartmann_3'
    def func(self,coord):
        c = array([1, 1.2, 3, 3.2])
        A = array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        P = array([[0.3689, 0.1170, 0.2673], 
                   [0.4699, 0.4387, 0.747], 
                   [0.1091, 0.8732, 0.5547],
                   [0.0382, 0.5743, 0.8828]])
        out = sum(c * exp(-sum(A * (repmat(coord, 4, 1) - P) ** 2, axis = 1)))
        return out

class Synthetic:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-3, 3), 'y': (-3, 3)}
        self.name='Synthetic'
    def func(self, coord):
        # setting up variables for the Synthetic function
        X,Y=synthetic_input()
        coord=np.array(coord)
        kern_syn = GPy.kern.RBF(2, variance=1, lengthscale=(0.3, 0.3), ARD=True)
        gp_syn = GPy.models.GPRegression(X, Y, kern_syn)
        gp_syn.optimize()
        if len(coord.shape) == 1:
            _coord = coord[np.newaxis, :]
            out = gp_syn.predict_noiseless(_coord)[0][0][0]
        elif len(coord.shape) == 3 :
            _coord = coord.reshape(coord.shape[0], -1).T
            out = gp_syn.predict_noiseless(_coord)[0].reshape(coord.shape[1],-1)
        else:
            _coord = coord
            out = gp_syn.predict_noiseless(_coord)[0][0][0]
        return out

class Hartmann_6:
    def __init__(self):
        self.input_dim=6
        self.bounds={'x': (0, 1), 'y': (0, 1),'z': (0, 1), 'a': (0, 1),'b': (0, 1), 'c': (0, 1)}
        self.name='Hartmann_6'
    def func(self,coord):
        coord=np.array(coord)
        xx=coord
        if len(xx.shape) == 1:
            xx = xx.reshape((1, 6))

        assert xx.shape[1] == 6

        n = xx.shape[0]
        y = np.zeros(n)
        for i in range(n):
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])
            P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])

            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = xx[i, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2

                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            y[i] = -outer
        return -y[0]

class Ackley:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-32.768, 32.768), 'y': (-32.768, 32.768)}
        self.name='Ackley'
    def func(self,coord):
        firstSum = 0.0
        secondSum = 0.0
        for c in coord:
            firstSum += c**2.0
            secondSum += math.cos(2.0*math.pi*c)
        n = float(len(coord))
        return 20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
   
class Rosenbrock:
    def __init__(self):
        self.input_dim=6
        self.bounds={'x': (-5, 10), 'y':  (-5, 10), 'z': (-5, 10),'a': (-5, 10),'b': (-5, 10),'c': (-5, 10)}
        self.name='Rosenbrock'
    def func(self,coord):
        return -1 * sum(100.0*(coord[1:]-coord[:-1]**2.0)**2.0 + (1-coord[:-1])**2.0)


class Ackley_6:
    def __init__(self):
        self.input_dim=6
        self.bounds={'x': (-32.768, 32.768), 'y': (-32.768, 32.768),'z': (-32.768, 32.768),'a': (-32.768, 32.768),'b': (-32.768, 32.768),'c': (-32.768, 32.768)}
        self.name='Ackley_6'
    def func(self,coord):
        firstSum = 0.0
        secondSum = 0.0
        for c in coord:
            firstSum += c**2.0
            secondSum += math.cos(2.0*math.pi*c)
        n = float(len(coord))
        return 20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e


class Schwefel:
    def __init__(self):
        self.input_dim=5
        self.bounds={'x': (-500, 500), 'y': (-500, 500),'z': (-500, 500),'a': (-500, 500),'b':(-500, 500)}
        self.name='Schwefel'
    def func(self,coord):
        alpha = 418.982887
        fitness = 0
        for i in range(len(coord)):
            fitness -= coord[i]*math.sin(math.sqrt(math.fabs(coord[i])))
        return  - (float(fitness) + alpha*len(coord)) 

class Alpine:
    def __init__(self):
        self.input_dim=6
        self.bounds={'x': (0,10), 'y': (0,10),'z':(0,10),'a': (0,10),'b':(0,10),'c':(0,10)}
        self.name='Alpine'
    def func(self,coord):
        fitness = 0
        for i in range(len(coord)):
            fitness += math.fabs(0.1*coord[i]+coord[i]*math.sin(coord[i]))
        return  - fitness

class Mixture:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (0,1), 'y': (0,1)}
        self.name='Mixture'
        self.peaks=9
    def func(self,coord):
        y=2*multivariate_normal.pdf(coord,mean=[0.5,0.5],cov=0.07*np.eye(2))
        if self.peaks>=2:
            y+=1.8*multivariate_normal.pdf(coord,mean=[0.2,0.2],cov=0.03*np.eye(2))
        if self.peaks>=3:
            y+=1.7*multivariate_normal.pdf(coord,mean=[0.7,0.7],cov=0.07*np.eye(2))
        if self.peaks>=4:
            y+=1*multivariate_normal.pdf(coord,mean=[0.8,0.5],cov=0.02*np.eye(2))
        if self.peaks>=5:
            y+=1.7*multivariate_normal.pdf(coord,mean=[0.4,0.6],cov=0.005*np.eye(2))
        if self.peaks>=6:
            y+=1.75*multivariate_normal.pdf(coord,mean=[0.3,0.4],cov=0.0012*np.eye(2))
        if self.peaks>=7:
            y+=1.75*multivariate_normal.pdf(coord,mean=[0.9,0.8],cov=0.01*np.eye(2))
        if self.peaks>=8:
            y+=1.75*multivariate_normal.pdf(coord,mean=[0.2,0.6],cov=0.01*np.eye(2))
        if self.peaks>=9:
            y+=1.75*multivariate_normal.pdf(coord,mean=[0.9,0.3],cov=0.01*np.eye(2))
        return y

class Eggholder:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (- 512,512), 'y': (- 512,512)}
        self.name='Eggholder'
    def func(self,coord):
        if len(coord.shape)==1:
            x1=coord[0]
            x2=coord[1]
        else:
            x1=coord[:,0]
            x2=coord[:,1]
            
        func_val = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47))) + -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))
        return - func_val

class Michalewicz:
    def __init__(self):
        self.input_dim=10
        self.bounds={'x': (0,math.pi), 'y': (0,math.pi),'z': (0,math.pi),'a': (0,math.pi),'b': (0,math.pi),'c': (0,math.pi),'d': (0,math.pi),'e': (0,math.pi),'f': (0,math.pi),'g': (0,math.pi)}
        self.name='Michalewicz'
    def func(self,coord):
        n = len(coord)
        j = np.arange( 1., n+1 )
        return  sum( np.sin(coord) * np.sin( j * coord**2 / math.pi ) ** (2 * 10) )

class Dropwave:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-5.12,5.12), 'y': (-5.12,5.12)}
        self.name='Dropwave'
    def func(self,coord):
        if len(coord.shape)==1:
            x1=coord[0]
            x2=coord[1]
        else:
            x1=coord[:,0]
            x2=coord[:,1]
        
        fval = - (1+np.cos(12*np.sqrt(x1**2+x2**2))) / (0.5*(x1**2+x2**2)+2) 
        return - fval

class Shekel:
    def __init__(self):
        self.input_dim=4
        self.bounds={'x': (0,10), 'y': (0,10),'z': (0,10),'a': (0,10)}
        self.name='Shekel'
    def func(self,coord):
        m=5
        C = [[4, 1 ,8 ,6,3],[4, 1 ,8 ,6,7], [4, 1 ,8 ,6,3], [4, 1 ,8 ,6,7]]
        B = [0.1, 0.2, 0.2, 0.4, 0.4]
        outer=0
        for i in range (m):
            inner=0
            for j in range(self.input_dim):
                inner=inner+ (coord[j]-C[j][i])**2
            outer=outer+(1/(inner+B[i]))
        
        return(outer)

class sin:
    def __init__(self,noisy=True):
        self.input_dim=1
        self.bounds={'x':(-1,15)}
        #self.bounds={'x':(0,1)}
        self.name='sin'
        self.noisy=noisy
        self.noise_std=0.05
    def func(self,coord):
        x=np.asarray(coord)
        fval=np.sin(x)
        if self.noisy:
            return fval + np.random.normal(0, self.noise_std, size=(x.shape[0], ))
        else:
            return fval

class cos:
    def __init__(self):
        self.input_dim=1
        self.bounds={'x':(-1,15)}
        #self.bounds={'x':(0,1)}
        self.name='sin'
    def func(self,coord):
        x=np.asarray(coord)
        fval=np.cos(x)
        return fval

class sincos:
    def __init__(self):
        self.input_dim=2
        self.bounds={'x': (-4, 4), 'y': (-4,4 )}
        self.name='sinxy'
    def findSdev(self):
        num_points_per_dim=100
        bounds=self.bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
            arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds=np.asarray(bounds)
        X=np.array([np.random.uniform(x[0], x[1], size=num_points_per_dim) for x in arr_bounds])
        X=X.reshape(num_points_per_dim,-1)
        y=self.func_noisless(X)
        sdv=np.std(y)
        return sdv

    def func(self,coord):
        if(coord.ndim==1):
            coord=coord[np.newaxis,:]
        X1=coord[:,0]
        X2=coord[:,1]
        n=coord.shape[0]
        std=0.1*self.findSdev()
        noise = np.random.normal(0,std,n).reshape(n,1)
        out =  np.sin(X1)*np.cos(X2)+noise
        out=np.squeeze(out)
        return out

    def func_noisless(self,coord):
        if(coord.ndim==1):
            coord=coord[np.newaxis,:]
        X1=coord[:,0]
        X2=coord[:,1]
        n=coord.shape[0]
        out =   np.sin(X1)*np.cos(X2)
        out=np.squeeze(out)
        return out 
    
    





