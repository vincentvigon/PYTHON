from joblib import Parallel, delayed
import time
import numpy as np
from multiprocessing import cpu_count
print("nb cpu avaiable:",cpu_count())
import scipy.optimize as sco
import matplotlib.pyplot as plt
import tensorflow as tf

def l2norm(x):
    return np.linalg.norm(x)


def one_opt(fn,init):
    options = {'maxiter': 2000, 'disp': False, 'tol_norm': l2norm, 'ftol': 1e-6, 'fatol': 1e-6,
        'jac_options': {'inner_maxiter': 1000}}
    sol_ref = sco.root(fn, init, method='krylov', options=options)
    return sol_ref.x,sol_ref.nit


def several_opt(fns,inits,n_jobs):
    results = Parallel(n_jobs=n_jobs)(delayed(one_opt)(fn, init) for fn,init in zip(fns,inits))
    return results

def several_opt_seq(fns,inits):
    results=[]
    for fn, init in zip(fns, inits):
        result=one_opt(fn,init)
        results.append(result)
    return results


class PseudoAgentNewton:
    def call_model(self,X):
        return X+tf.random.uniform(minval=-0.1,maxval=0.1,shape=X.shape)

def mse(a):
    return tf.reduce_mean(tf.square(a))


class PseudoDataNewton:

    def __init__(self,nb_jobs):
        self.nb_jobs=nb_jobs

    def make_XY(self,batch_size):
        nx=100
        x=tf.linspace(0.,1.,nx)[None,:]
        nus=tf.random.uniform(minval=2,maxval=10,shape=[batch_size,1])
        X=tf.sin(nus*x)[:,:,None] #(batch_size,nx,1)
        Y=X #(batch_size,nx,1)
        return X,Y

    def score(self, agent) -> dict:
        batch_size=100#todo
        X,Y=self.make_XY(batch_size)
        Y_pred=agent.call_model(X)

        fns=[]
        targets=[]
        inits=[]
        for i in range(batch_size):
            target=X[i,:,0]
            init=Y_pred[i,:,0]
            fn=self.get_G_for_scipy(target)
            targets.append(target)
            fns.append(fn)
            inits.append(init)

        results=several_opt(fns,inits,self.nb_jobs)
        nb_it=0
        error=0
        for i in range(batch_size):
            error+=mse(targets[i]-results[i][0])
            nb_it+=results[i][1]
        nb_it/=batch_size
        error/=batch_size

        return {"nb_it":nb_it,"error":error}

    #fonction par batch
    def G(self,U,target):
        return (U-target)**2

    #fonction non-batchée
    def get_G_for_scipy(self,target_i):
        #scipy veut une fonction non batchée
        return lambda U_i:self.G(U_i,target_i[None, :])[0, :]



def test_pseudo_newton():

    def one(nb_jobs):
        pseudo_agent=PseudoAgentNewton()
        pseudo_data=PseudoDataNewton(nb_jobs)
        ti0=time.time()
        scores=pseudo_data.score(pseudo_agent)
        duration=time.time()-ti0
        print("scores",scores)
        print("duration",duration)
        return duration

    nb_jobs=[1,4,8,12,16]
    durations=[]
    for nb_job in nb_jobs:
        durations.append(one(nb_job))

    plt.plot(nb_jobs,durations)
    plt.show()






class ParamFn:
    def __init__(self,nx):
        self.x=np.linspace(0,1,nx)
    def give_fn_for_scipy(self,nu):
        target=np.sin(nu*self.x)
        return lambda U: (U - target) ** 2


def test_several_opt():
    def mse(a):
        return np.mean(np.square(a))
    nx = 10
    param_fn=ParamFn(nx)
    x = np.linspace(0, 1, nx)

    fns=[]
    inits=[]
    targets=[]
    nus=[4,7,9]

    for nu in nus:
        target = np.sin(nu * x)
        fn = param_fn.give_fn_for_scipy(nu)
        init = target + np.random.uniform(-0.1, 0.1, nx)
        fns.append(fn)
        inits.append(init)
        targets.append(target)

    ti0 = time.time()

    results=several_opt(fns, inits,16)
    duration = time.time()-ti0
    print("duration",duration)

    nits=[]
    for result_nit,target in zip(results,targets):
        result=result_nit[0]
        nits.append(result_nit[1])
        print("mse",mse(result-target))
        plt.plot(result,"+")
        plt.plot(target)

    print(nits)
    #
    # for fn in fns:
    #     plt.plot(fn(x))

    plt.show()


#scipy_test()
#est_paral()
#test_several_opt()
test_pseudo_newton()










