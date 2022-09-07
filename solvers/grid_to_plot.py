from matplotlib import cm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


pp=print

class Grid_to_plot:

    def __init__(self, nx,xmin,xmax,nt,tmax):

        self.nb_t = nt
        self.max_t = tmax
        self.nb_x = nx
        self.min_x =xmin
        self.max_x = xmax


        # un zero en float32
        zero = tf.constant(0.)
        # pour produire un linspace en float32
        self.t = tf.linspace(zero, self.max_t, self.nb_t)
        self.x = tf.linspace(self.min_x, self.max_x, self.nb_x)

        # vecteur répétés.
        #   ATTENTION À L'ORDRE
        self.xx,self.tt  = tf.meshgrid( self.x ,self.t)

        self.t_=tf.reshape(self.tt,[-1])
        self.x_=tf.reshape(self.xx,[-1])
        self.tx_=tf.stack([self.t_,self.x_],axis=1)

    def to_mat(self, tx):
        return tf.reshape(tx, [self.nb_t, self.nb_x])

    def plot_2d(self, ax, U):

        if len(U.shape)==1:
            U=self.to_mat(U)

        im=ax.pcolormesh(self.xx,self.tt,U,shading='auto')
        #im =ax.imshow(U, origin="lower", cmap="jet", extent=[0, self.max_x, 0, self.max_t])

        #ax.set_xlabel("x")
        #ax.set_ylabel("t")

        return im

    def plot_several_times(self,ax, U ,initial=None):
        nb = 20

        if len(U.shape)==1:
            U=self.to_mat(U)

        if initial is not None:
            ax.plot(self.x ,initial ,"k+" ,label="initial")

        for k in range(nb):
            i = int(k * self.nb_t / nb)
            label= np.round(self.t[i].numpy(), 2) if k==0 or k== nb-1 else None
            ax.plot(self.x, U[i, :], color=cm.jet(k / nb), label=label)
        ax.legend()


    def plot_several_spaces(self,ax, U):
        nb = 20

        if len(U.shape)==1:
            U=self.to_mat(U)

        for k in range(nb):
            i = int(k * self.nb_x / nb)
            label= np.round(self.x[i].numpy(), 2) if k==0 or k== nb-1 else None
            ax.plot(self.t, U[:, i], color=cm.jet(k / nb), label=label)
        ax.legend()




def test_grid():
    nt=50
    nx = 400
    xmin = 0.
    xmax = 1.
    tmax = 0.1

    grid = Grid_to_plot(nx,xmin,xmax,nt,tmax)

    func=lambda tx : tx[:,0]

    U = func(grid.tx_)

    fig,ax=plt.subplots()
    grid.plot_several_times(ax,U)
    plt.show()

    fig,ax=plt.subplots()
    grid.plot_2d(ax, U)
    plt.show()

if __name__=="__main__":
    test_grid()

