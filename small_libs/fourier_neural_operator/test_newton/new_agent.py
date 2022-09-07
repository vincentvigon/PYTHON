from fourier_neural_operator.FNO_1d_plus import FNO1d_plus
from grid_up.grid_up import GridUp_dataMaker, GridUp_agent

pp=print
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
#import fourier_neural_operator.test_env as te
from typing import List
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
print("nb cpu avaiable:",cpu_count())
import scipy.optimize as sco


class Mesh:
  def __init__(self, N, a, b,dtype):
    self.N = N
    self.a = a
    self.b = b
    self.m = tf.linspace(tf.constant(a,dtype=dtype),tf.constant(b,dtype=dtype),N)
    self.h= tf.abs(self.m[1]-self.m[0])
    self.mid = (self.a + self.b) / 2
    self.L = self.b - self.a


def l2norm(x):
    return np.linalg.norm(x)


def one_opt(fn,init):
    options = {'maxiter': 2000, 'disp': True, 'tol_norm': l2norm, 'ftol': 1e-6, 'fatol': 1e-6,
        'jac_options': {'inner_maxiter': 1000}}
    sol_ref = sco.root(fn, init, method='krylov', options=options)
    return sol_ref.x,sol_ref.nit


class NewtonData(GridUp_dataMaker):

    #for gridup
    def score(self, agent:'AgentNewton') -> dict:
        res_type=0
        batch_size=1
        X,Y=self.make_XY(batch_size)
        Y_pred = agent.call_model(X)

        X,Y,Y_pred=np.array(X,np.float64),np.array(Y,np.float64),np.array(Y_pred,np.float64)
        f=X[:,:,0]
        alpha=X[:,:,1]
        G=self.get_G_for_scipy(f,alpha)

        init=Y_pred[:,:,0]
        res,nit=one_opt(G,np.ones_like(init))


        print(nit)

        return None

    #for gridup
    def plot_prediction(self, ax, agent:'AgentNewton',custom_arg=None) -> None:
        nb=10
        tf.random.set_seed(123)
        X,Y=self.make_XY(nb)
        Y_pred=agent.call_model(X)
        i=custom_arg.get('i',0)
        U=Y[:,:,0]
        U_pred = Y_pred[:, :, 0]
        if custom_arg.get("U",False):
            ax.plot(U[i],label="U true")
            ax.plot(U_pred[i],label="U pred")
        elif custom_arg.get("residues",False):
            residues=self.G2(X,Y)
            residues_pred=self.G2(X,Y_pred)
            ax.plot(residues[i], label="residues true")
            ax.plot(residues_pred[i], label="residues pred")

    # for gridup
    @tf.function
    def make_XY(self, batch_size):
        if self.verbose:
            print("traçage de la fonction generate_XY")

        alpha = self.generate_alpha(batch_size)
        U = self.generate_alpha(batch_size)
        f_nor = self.elliptic(U, alpha) / self.normalisation_for_f
        X = tf.stack([f_nor, alpha], axis=2)
        Y = U[:, :, None]
        return X, Y

    def __init__(self,
            a, #deb de l'intervalle
            b, #fin de l'intervalle
            N, #nombre de points
            k, #fonction déterminant l'équation
            kind="fourier", # "fourier" ou "gauss"
            BC="neumann",# "dirichlet" ou "neumann"
            dtype=tf.float32,
            verbose=True):
        self.a=a
        self.b=b
        self.N=N
        self.k=k
        self.kind=kind
        self.BC=BC
        self.dtype = dtype
        self.verbose = verbose
        self.n_jobs=1

        self.mesh = Mesh(N, self.a, self.b,self.dtype)
        # noinspection PyUnresolvedReferences
        self.dist = tfp.distributions.Normal(loc=tf.constant(0.,dtype=self.dtype), scale=tf.constant(1.,dtype=self.dtype) )
        #self.dist.dtype=self.dtype
        if self.verbose:
            print(f"NewtonData with: nb points={N}, kind of data:{kind}")
        self.normalisation_for_f=1000.

    def elliptic(self,U,alpha):
        # reaction term
        reac = alpha * U
        Dm, Dp=self.first_order_derivative(U)
        diffusion = self.diffusion(Dm,Dp)
        return diffusion+reac

    def diffusion(self,Dm,Dp):
        return (Dm - Dp) / self.mesh.h

    def first_order_derivative(self, U):
        h = self.mesh.h  # (N)

        Um = U[:,:-1]  # (N)   Um = U_i-1
        Up = U[:,1:]  # (N)  Up = U_i+1

        if self.BC == "dirichlet":
            Um = tf.concat([U[:, :1 ], Um], axis=1)
            Up = tf.concat( [Up,U[:, -1:]], axis=1)

        elif self.BC == "periodic":
            Um = tf.concat([U[:, -1:], Um], axis=1)
            Up = tf.concat([Up, U[:, :1]], axis=1)

        elif self.BC == "neumann":
            pass
            # DerLeft=(U[:,1]-U[:,0])/self.mesh.h
            # DerRight=(U[:,-1]-U[:,-2])/self.mesh.h
            # Um = tf.concat([DerLeft, Um], axis=1)
            # Up = tf.concat([Up, DerRight], axis=1)
        else:
            raise Exception(f"must be 'dirichlet' or 'periodic': 'neumann' will come soon, found:{self.BC}")

        # diffusion terms
        Kp = 0.5 * (self.k(U) + self.k(Up))
        Km = 0.5 * (self.k(Um) + self.k(U))

        Dp=Kp * (Up - U)/h
        Dm=Km * (U - Um)/h

        return Dm,Dp


    def elliptic_scipy(self,U,alpha):
        reac = alpha*U
        Um = U[:,:-1]  # (N)   Um = U_i-1
        Up = U[:,1:]  # (N)  Up = U_i+1

        if self.BC == "dirichlet":
            Um = np.concatenate([U[:, :1 ], Um],axis=1)
            Up = np.concatenate( [Up,U[:,-1:]],axis=1)

        elif self.BC == "periodic":
            Um = np.concatenate([U[:, -1:], Um],axis=1)
            Up = np.concatenate([Up, U[:,:1]],axis=1)

        elif self.BC == "neumann":
            pass
            # DerLeft=(U[:,1]-U[:,0])/self.mesh.h
            # DerRight=(U[:,-1]-U[:,-2])/self.mesh.h
            # Um = tf.concat([DerLeft, Um], axis=1)
            # Up = tf.concat([Up, DerRight], axis=1)
        else:
            raise Exception(f"must be 'dirichlet' or 'periodic': 'neumann' will come soon, found:{self.BC}")

        # diffusion terms
        Kp = 0.5 * (self.k(U) + self.k(Up))
        Km = 0.5 * (self.k(Um) + self.k(U))

        Dp=Kp * (Up - U)/self.mesh.h
        Dm=Km * (U - Um)/self.mesh.h
        diffusion = (Dm - Dp) / self.mesh.h

        return diffusion+reac

    def get_G_for_scipy(self,f,alpha):
        #scipy veut une fonction non batchée
        return lambda U: self.elliptic_scipy(U,alpha)-f

    def G(self, U,f,alpha):
        ellip = self.elliptic(U,alpha)
        res = ellip - f
        return res

    def G2(self,X,Y):
        f_nor=X[:,:,0]
        f=f_nor*self.normalisation_for_f
        alpha=X[:,:,1]
        U=Y[:,:,0]
        return self.G(U,f,alpha)

    def generate_fourier(self,nb_data):
        nb_fourier=6
        n = tf.range(1,nb_fourier+1,dtype=self.dtype)[:,None,None]
        an = tf.random.uniform(minval=-1,maxval=1,shape=[nb_fourier,nb_data,1],dtype=self.dtype)
        an/=n #pour que les hautes fréquences soient moins présentes

        nu = 2 * np.pi / self.mesh.L * n

        a0 = tf.random.uniform(minval=-1,maxval=1,shape=[1,nb_data,1],dtype=self.dtype)
        t=self.mesh.m[None,None,:]
        res= a0 + an*tf.sin(t* nu)
        res=tf.reduce_sum(res,axis=0)
        return res

    def generate_gaussian_mix(self,nb_data):
        nb_gauss_max=6

        std = tf.random.uniform(minval=0.025, maxval=0.07,shape=[nb_gauss_max,nb_data,1],dtype=self.dtype)
        mean= tf.random.uniform(minval=self.mesh.mid - 0.25 * self.mesh.L , maxval=self.mesh.mid + 0.25 * self.mesh.L,shape=[nb_gauss_max,nb_data,1],dtype=self.dtype)
        x = (self.mesh.m[None,None,:]-mean)/std

        gauss=self.dist.prob(x)
        mask=tf.random.uniform(minval=0,maxval=1.75,shape=[nb_gauss_max-1,nb_data,1],dtype=self.dtype)
        mask=tf.math.floor(mask) #un peu plus de 0 que de 1 dans ce masque

        mask=tf.concat([tf.ones([1,nb_data,1]),mask],axis=0) #pour avoir au moins une gaussienne à chaque fois
        gauss*=mask
        res=tf.reduce_sum(gauss,axis=0) #on somme les gaussiennes

        res = res / (tf.reduce_max(res,axis=1)[:,None]+0.1) + 0.5
        return res

    def generate_alpha(self,nb_data):
        if self.kind=="gauss":
            return self.generate_gaussian_mix(nb_data)
        elif self.kind=="fourier":
            return self.generate_fourier(nb_data)
        else:
            raise Exception(f"kind must be 'gauss' or 'fourier'. Found:{self.kind}")


    def plot_data(self,X,Y):
        f_nor=X[:, :, 0]
        f = f_nor*self.normalisation_for_f
        alpha=X[:, :, 1]
        U=Y[:, :, 0]
        g=self.G(U,f,alpha)
        nb=5
        fig, ax = plt.subplots(nb, 4, figsize=(16, nb))

        for i in range(nb):
            ax[i,0].plot(self.mesh.m,f_nor[i,:] )
            ax[i,1].plot(self.mesh.m, alpha[i,:])
            ax[i,2].plot(self.mesh.m, U[i,:] )
            ax[i,3].plot(self.mesh.m, g[i,:])
        ax[0, 0].set_title("f_nor")
        ax[0, 1].set_title("alpha")
        ax[0, 2].set_title("U")
        ax[0, 3].set_title("G(U)")
        fig.tight_layout()

    # noinspection PyUnboundLocalVariable
    def losses_fn(self, X, Y, Y_pred, keys,coef_for_derivative):
        U = Y[:, :, 0]
        U_pred = Y_pred[:, :, 0]
        alpha = X[:, :, 1]
        result = {}

        if "D" in keys or "diffusion" in keys or "residues" in keys:
            Dm, Dp = self.first_order_derivative(U)
            Dm_pred, Dp_pred = self.first_order_derivative(U_pred)

        if "diffusion" in keys or "residues" in keys:
            diffusion = self.diffusion(Dm, Dp)
            diffusion_pred = self.diffusion(Dm_pred, Dp_pred)

        if "residues" in keys:
            f = X[:, :, 0]*self.normalisation_for_f
            reac_pred = alpha * U_pred
            residues_pred = diffusion_pred + reac_pred - f

        if "U" in keys:
            diff_pond=(U - U_pred)*(alpha+1) #pour appuyer là où c'est important
            result["U"] = mse(diff_pond)
        if "D" in keys:
            result["D"] = (mse(Dm - Dm_pred) + mse(Dp - Dp_pred))*coef_for_derivative
        if "diffusion" in keys:
            result["diffusion"] = mse(diffusion - diffusion_pred)*coef_for_derivative**2
        if "residues" in keys:
            result["residues"] = mse(residues_pred)*coef_for_derivative**2

        return result

    #
    # def one_opt(self,Xloc,gn,init_type,agent:'AgentNewton'):
    #   def l2norm(x):
    #     return np.linalg.norm(x)
    #
    #   print("coucou")
    #   start_time = time.time()
    #   options = {'maxiter': 200, 'disp': True, 'tol_norm': l2norm, 'ftol': 1e-6, 'fatol': 1e-6,
    #       'jac_options': {'inner_maxiter': 200}}
    #   if init_type==0:
    #     U0=np.ones_like(Xloc[:,0],dtype=np.float64)
    #     sol_ref = sco.root(gn, U0, method='krylov', options=options)
    #   else:
    #     U0=agent.call_model(Xloc)
    #     sol_ref = sco.root(gn, U0, method='krylov', options=options)
    #   if sol_ref.success:
    #     niter = sol_ref.nit
    #   else:
    #     niter = 2000
    #   print("iinn",niter)
    #   basic = time.time()-start_time
    #   return sol_ref.x,niter,time


def mse(a):
    return tf.reduce_mean(tf.square(a))


class AgentNewton(GridUp_agent):

    #for gridup (for early stopping)
    def get_weights(self) -> List[tf.Tensor]:
        return self.model.get_weights()

    #for gridup (for early stopping)
    def set_weights(self, weights: List[tf.Tensor]) -> None:
        self.model.set_weights(weights)

    #for gridup (for early stopping)
    def call_model(self,X):
        X=self.augment(X)
        return self.model.call(X)

    #for gridup (for early stopping)
    def train_step(self, data_maker: NewtonData):
        loss, _=self.train_step_with_details(data_maker)
        return loss

    def __init__(self,
            name_of_losses,
            modes,
            width,
            nb_layer,
            first_channel_unchanged,
            freq_mix_size,
            pad_prop,
            pad_kind,
            batch_size,
            only_one_optimizer,
            lr,
            augmentation_level
    ):

        self.model = FNO1d_plus(modes, width,1,nb_layer,first_channel_unchanged,freq_mix_size,pad_prop,pad_kind)
        self.optimizer = tf.keras.optimizers.Adam()
        self.batch_size = batch_size
        self.only_one_optimizer=only_one_optimizer
        self.name_of_losses=name_of_losses
        self.augmentation_level=augmentation_level

        if only_one_optimizer:
            self.optimizers=tf.keras.optimizers.Adam(lr)
        else:
            self.optimizers={name:tf.keras.optimizers.Adam(lr) for name in self.name_of_losses}


    def augment(self,X):
        if self.augmentation_level==0:
            return X
        elif self.augmentation_level==1:
            f_nor=X[:,:,0]
            nb_x=f_nor.shape[1]

            alpha=X[:,:,1]
            ff=tf.cumsum(f_nor,axis=1)/nb_x
            return tf.stack([f_nor,alpha,ff],axis=2)

        elif self.augmentation_level==2:
            f_nor=X[:,:,0]
            nb_x=f_nor.shape[1]
            alpha=X[:,:,1]
            ff=tf.cumsum(f_nor,axis=1)/nb_x
            fff = tf.cumsum(ff, axis=1) / nb_x
            return tf.stack([f_nor,alpha,ff,fff],axis=2)
        else:
            raise Exception(f"augmentation level must be in [0,1,2], found:{self.augmentation_level}")


    @tf.function
    def train_step_with_details(self, data_maker: NewtonData):
        X,Y=data_maker.make_XY(self.batch_size)
        X=self.augment(X)
        with tf.GradientTape(persistent=True) as tape:
            Y_pred=self.model.call(X)
            losses=data_maker.losses_fn(X,Y,Y_pred,self.name_of_losses,coef_for_derivative=0.01)
            loss=sum([loss_val for loss_val in losses.values()])

        tv=self.model.trainable_variables
        if self.only_one_optimizer:
            grad=tape.gradient(loss,tv)
            self.optimizer.apply_gradients(zip(grad,tv))
        else:
            for key in self.name_of_losses:
                grad=tape.gradient(losses[key],tv)
                self.optimizers[key].apply_gradients(zip(grad,tv))

        del tape
        return loss,losses


if __name__=="__main__":
    data=NewtonData(0,1,200,lambda u:1+u**4,"gauss",BC="dirichlet")
    agent=AgentNewton(["U"],20,20,4,False,5,0,"no_padding",64,True,1e-3,0)
    data.score(agent)


