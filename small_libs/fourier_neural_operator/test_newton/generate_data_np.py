import matplotlib.pyplot as plt
import numpy as np
from typing import *
import tensorflow as tf
import scipy.optimize as sco
import scipy as sci
import scipy.stats as stats
from scipy.interpolate import interp1d
import time
from matplotlib import cm
import random as rand


class Mesh:
  def __init__(self, N, a, b):
    self.N = N
    self.a = a
    self.b = b
    self.m = np.linspace(a,b,N)
    self.h= abs(self.m[1]-self.m[0])


  #
  # def remesh(self,N):
  #   self.N=N
  #   self.m = np.linspace(self.a,self.b,N)
  #   self.h= abs(self.m[1]-self.m[0])
  #
  # def __call__(self):
  #   return self.m


class Newton:  # à modifier /creation des jeux de données

    def __init__(self, N, a, b, k=lambda u: np.ones_like(u), kp=lambda u: np.zeros_like(u), nb_eq=1, nb_sol_by_eq=1,
            ubc=0.0, bc='Neumann',verbose=True):

        self.nb_data = nb_eq * nb_sol_by_eq
        self.nb_eq = nb_eq
        self.nb_sol_by_eq = nb_sol_by_eq
        self.k = k
        self.kp = kp
        self.bc = bc
        self.ubc = ubc
        self.N = N
        if self.bc == 'Neumann':
            self.f_type = 1
        else:
            self.f_type = 0
        self.mesh = Mesh(N, a, b)

        if verbose:
            print('====== Init ======')
            print('mesh size: ', self.mesh.N)
            print('mesh step: ', self.mesh.h)
            print('bc type  : ', self.bc)


    def elliptic(self, U,alpha_loc):
        """
        alpha U + nabla(k(U) nabla(U))
        ex: k(U) = 1 + U^4
        """
        h = self.mesh.h
        Up = np.roll(U, -1)  # Up = U_i+1
        Um = np.roll(U, 1)  # Um = U_i-1
        # reaction term
        reac = alpha_loc * U
        # BC
        if self.bc == 'Dirichlet':
            Um[0] = self.ubc
            Up[-1] = self.ubc

            # diffusion terms
            Kp = 0.5 * (self.k(U) + self.k(Up))
            Km = 0.5 * (self.k(Um) + self.k(U))
            diff = - (Kp * (Up - U) - Km * (U - Um)) / h ** 2

        elif self.bc == 'Neumann':
            # diffusion terms
            Kp = 0.5 * (self.k(U) + self.k(Up))
            Km = 0.5 * (self.k(Um) + self.k(U))

            diff = - (Kp * (Up - U) - Km * (U - Um)) / h ** 2
            diff[0] = - (Kp[0] * (Up[0] - U[0]) + self.ubc) / h ** 2  # - Km * (U - Um) = uL neumann on left boundary
            diff[-1] = - (-self.ubc - Km[-1] * (U[-1] - Um[-1])) / h ** 2  # - Kp * (Up - U) = uR neumann on right boundary
        else:
            raise Exception('Not yet implemented.')

        return reac + diff


    def get_G_for_scipy(self,f,alpha):
        return lambda U:self.G(U,f,alpha)


    def G(self, U,f,alpha):
        """
        Cacule les résidus:
         G(U)= alpha U + nabla(k(U) nabla(U)) - f
        """
        ellip = self.elliptic(U,alpha)
        res = ellip - f
        return res


    def generate_fourier(self, a0=0.0, f_type=0):
        if f_type == 0:
            nbFourierCoef = np.random.randint(1, 7)
            n = np.arange(1, nbFourierCoef + 1)
            an = np.random.uniform(-1, 1, nbFourierCoef) / n

            nu = 2 * np.pi / (self.mesh.b - self.mesh.a)
            res = a0 + np.sum(an * np.sin(np.outer(self.mesh.m, n) * nu), axis=1)
            self.ubc = a0
        else:
            nb_g = rand.randint(0, 6)
            x_0 = (self.mesh.a + self.mesh.b) / 2
            L = self.mesh.b - self.mesh.a
            std = np.random.uniform(0.025, 0.08)
            res = sci.stats.norm.pdf(self.mesh.m, np.random.uniform(x_0 - 0.25 * L, x_0 + 0.25 * L), std)
            for i in range(1, nb_g):
                std = np.random.uniform(0.025, 0.07)
                res = res + sci.stats.norm.pdf(self.mesh.m, np.random.uniform(x_0 - 0.25 * L, x_0 + 0.25 * L), std)
            res = res / np.max(res) + 0.5
        return res


    def generate_data_train(self):
        """"""
        """
        On génére 
          u,alpha
        On en déduit le second membre
          f 
        qui permet d'avoir un résidu nul
        """
        # construct random f,u,alpha and compute g(u)
        X = np.zeros((self.nb_data, self.mesh.N, 2))  # f , alpha
        Y = np.zeros((self.nb_data, self.mesh.N, 1))  # u
        for i in range(0, self.nb_eq):
            X[i, :, 1] = self.generate_fourier(a0=1.5, f_type=self.f_type)
            alpha = np.copy(X[i, :, 1])
            for k in range(0, self.nb_sol_by_eq):
                U = self.generate_fourier(a0=1.2, f_type=self.f_type)
                Y[i, :, 0] = U  # generate u
                X[i, :, 0] = self.elliptic(U,alpha)  # generate f

        print(">>> end generate >>>")
        return X,Y


    def plot_data(self,X,Y, i):
        fig, ax = plt.subplots(2, 2, figsize=(16, 6))
        f=X[i, :, 0]
        alpha=X[i, :, 1]
        u=Y[i, :, 0]
        g=self.G(Y[i, :, 0],f,alpha)
        ax[0, 0].plot(self.mesh.m,f )
        ax[0, 1].plot(self.mesh.m, alpha)
        ax[1, 0].plot(self.mesh.m, u )
        ax[1, 1].plot(self.mesh.m, g)
        ax[0, 0].set_title("f")
        ax[0, 1].set_title("alpha")
        ax[1, 0].set_title("u")
        ax[1, 1].set_title("G(u)")
        fig.tight_layout()




def test():
    ku2 = lambda u: u ** 4 + 1.0  # non linéarité
    kpu2 = lambda u: 4.0 * u * u * u
    nb_data=1500

    Newton_train = Newton(N=256, a=0.0, b=1.0, k=ku2, kp=kpu2, nb_eq=nb_data, nb_sol_by_eq=1)

    ti0=time.time()
    X_train,Y_train=Newton_train.generate_data_train()
    duration=time.time()-ti0
    print(f"duration={duration} for nb_data={nb_data} with nb points={Newton_train.mesh.N}")


    print("X_train",X_train.shape)
    print("Y_train",Y_train.shape)

    Newton_train.plot_data(X_train,Y_train,0)
    plt.show()




test()

