import time
import numpy as np
import scipy.optimize as sco
import tensorflow as tf
pp=print
from small_libs.tf_or_np.backend import K


def l2norm(x):
    return np.linalg.norm(x)



def one_opt(fn,init):
    options = {'maxiter': 2000, 'disp': False, 'tol_norm': l2norm, 'ftol': 1e-6, 'fatol': 1e-6,
        'jac_options': {'inner_maxiter': 1000}}
    sol_ref = sco.root(fn, init, method='krylov', options=options)
    return sol_ref.x,sol_ref.nit



def mse_fn(a):
    return np.mean(np.square(a))



def test_one_opt():
    shape=[4,5]
    target=np.random.uniform(-1,1,size=shape)
    init=target+np.random.uniform(-0.1,0.1,size=shape)

    fn=lambda u:(u-target)**2

    x,nit=one_opt(fn,init)

    error=mse_fn(x-target)
    print(f"error:{error},nit:{nit}")



class PseudoAgentNewton:
    def call_model(self,X):
        return X**2+tf.random.uniform(minval=-0.1,maxval=0.1,shape=X.shape)

def mse(a):
    return tf.reduce_mean(tf.square(a))



class PseudoDataNewton:

    def __init__(self):
        pass

    def make_XY(self,batch_size):
        nx=200
        x=tf.linspace(0.,1.,nx)[None,:]
        nus=tf.random.uniform(minval=2,maxval=10,shape=[batch_size,1])
        X=tf.sin(nus*x)[:,:,None] #(batch_size,nx,1)
        Y=X**2
        return X,Y

    #fonction par batch #todo: important tout en numpy
    def G_np(self,U,X):
        U=np.concatenate([U[:,:1],U[:,1:]],axis=1)
        return (U - X**2) **2


    #fonction par batch
    def get_G_for_scipy(self,X):
        #scipy veut une fonction non batchée
        return lambda U:self.G_np(U,X)


    def score(self, agent) -> dict:
        batch_size=64
        X,Y=self.make_XY(batch_size)
        Y_pred=agent.call_model(X)
        X,Y,Y_pred=np.array(X),np.array(Y),np.array(Y_pred)

        X,Y=X[:,:,0],Y[:,:,0]
        Y_pred=Y_pred[:,:,0]

        fn=self.get_G_for_scipy(X)

        print("solution parfaite:",np.mean(fn(Y)))
        print("solution imparfaite:",np.mean(fn(Y_pred)))

        res,nit=one_opt(fn,Y_pred)
        error=mse_fn(res-Y)

        print(f"error:{error},nit:{nit}")

        return {"nb_it":nit,"error":error}


def test_pseudo_newton():
    agent=PseudoAgentNewton()
    data=PseudoDataNewton()
    ti0=time.time()
    data.score(agent)
    duration=time.time()-ti0
    print("duration",duration)



def first_order_derivative(U,h,BC,k,tn:K):

    Um = U[:,:-1]  # (N)   Um = U_i-1
    Up = U[:,1:]  # (N)  Up = U_i+1

    if BC == "dirichlet":
        Um = tn.concatenate([U[:, :1 ], Um], axis=1)
        Up = tn.concatenate( [Up,U[:, -1:]], axis=1)

    elif BC == "periodic":
        Um = tn.concatenate([U[:, -1:], Um], axis=1)
        Up = tn.concatenate([Up, U[:, :1]], axis=1)

    elif BC == "neumann":
        pass
        # DerLeft=(U[:,1]-U[:,0])/self.mesh.h
        # DerRight=(U[:,-1]-U[:,-2])/self.mesh.h
        # Um = tf.concat([DerLeft, Um], axis=1)
        # Up = tf.concat([Up, DerRight], axis=1)
    else:
        raise Exception(f"must be 'dirichlet' or 'periodic': 'neumann' will come soon, found:{self.BC}")

    # diffusion terms
    Kp = 0.5 * (k(U) + k(Up))
    Km = 0.5 * (k(Um) + k(U))

    Dp=Kp * (Up - U)/h
    Dm=Km * (U - Um)/h

    return Dm,Dp




def test_first_order_derivative():

    batch_size=5
    nx=200

    def one(tn):
        u = tn.linspace_float(0,1,nx)[None,:]
        a= tn.arange_float(0,batch_size)[:,None]
        U=a*u

        k= lambda u:1+u**4
        h=1e-3
        BC="dirichlet"
        Dm,Dp=first_order_derivative(U,h,BC,k,tn)
        return Dm,Dp

    tf_32=K("np",64)
    Dm_tf, Dp_tf = one(tf_32)

    np_32 = K("tf", 64)
    Dm_np, Dp_np = one(np_32)

    mae_fn= lambda a:np.mean(np.abs(a))

    print(mae_fn(Dm_tf-Dm_np))
    assert mae_fn(Dm_tf-Dm_np)<1e-6
    assert mae_fn(Dp_tf-Dp_np)<1e-6


if __name__=="__main__":
    #test_one_opt()
    test_pseudo_newton()
    #test_first_order_derivative()



