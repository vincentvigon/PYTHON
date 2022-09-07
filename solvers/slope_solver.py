import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from solvers.grid_to_plot import Grid_to_plot

D_fn_rusanov = lambda a, b: tf.maximum(tf.abs(a), tf.abs(b)) / 2


def pad_w(W, BC:str, p: int):
    if BC == "periodic":
        return periodic_padding(W, p)
    elif BC == "neumann":
        return neumann_padding(W, p)
    else:
        raise Exception("cette Boundary Condition  est inconnue:" + BC)


def periodic_padding(W, pad: int):
    left = W[:, :pad, :]
    right = W[:, -pad:, :]
    return tf.concat([right, W, left], axis=1)


def neumann_padding(W, pad):
    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = tf.ones([s[0], pad, s[2]]) * left_value[:, tf.newaxis, :]
    right_value_repeat = tf.ones([s[0], pad, s[2]]) * right_value[:, tf.newaxis, :]

    return tf.concat([left_value_repeat, W, right_value_repeat], axis=1)



def compute_numerical_flux_Rusanov(w_a: tf.Tensor,w_b: tf.Tensor,flux_fn) -> tf.Tensor:
    """"""
    """
    :param w_a: valeur à gauche, shape (b,nx+1,d)
    :param w_b: valeur à droite, shape (b,nx+1,d)
    :param flux_fn: flux de l'équation, ex: w_->w_**2/2 pour Burger
    :return: Le flux numérique
    """
    F_mean = (flux_fn(w_a) + flux_fn(w_b)) / 2 # nx+1
    D = D_fn_rusanov(w_a, w_b)  # nx+1
    F_num = F_mean - D * (w_b - w_a)

    return F_num # nx+1



def compute_w_at_interface(w,L,BC):
    w_ = pad_w(w, BC, 1)  # nx+2
    w_L = w_ - L  # nx+2
    w_R = w_ + L  # nx+2
    w_a, w_b = w_R[:, :-1, :], w_L[:, 1:, :]  # nx+1
    return w_a,w_b # nx+1


def minmod(a, b):
    """ minmod limiter """
    c1 = tf.logical_and(a > 0, b > 0)
    c2 = tf.logical_and(a < 0, b < 0)
    limiter = tf.where(c1, tf.minimum(a, b), tf.zeros_like(a))
    limiter = tf.where(c2, tf.maximum(a, b), limiter)
    return limiter

def one_time_step(Fnum, dt_over_dx: float, w: tf.Tensor) -> tf.Tensor:
    def time_step(w) -> tf.Tensor:
        dFnum = Fnum[:, 1:, :] - Fnum[:, :-1, :]
        return w - dt_over_dx * dFnum

    w_t1 = time_step(w)
    w_t2 = time_step(w_t1)
    return (w_t2 + w) / 2



class SolverFromL:

    def __init__(self, nx,dx,dt,BC,flux_fn):
        self.dt = dt
        self.nx = nx
        self.dx = dx
        self.BC=BC
        self.flux_fn=flux_fn

        self.dt_over_dx=dt/dx


    def compute_L(self,w):
        w_ = pad_w(w, self.BC, 2)  # nx+4
        dw = (w_[:, 1:, :] - w_[:, :-1, :]) / self.dx  # nx+ 3
        L = self.dx / 2 * minmod(dw[:, :-1, :], dw[:, 1:, :])  # nx+2
        return L  # nx+2

    def one_time_step(self,w):
        L=self.compute_L(w) #nx+2
        w_a, w_b=compute_w_at_interface(w,L,self.BC)
        Fnum = compute_numerical_flux_Rusanov(w_a,w_b,self.flux_fn)
        w_next=one_time_step(Fnum,self.dt_over_dx,w)
        return w_next,Fnum,L #nx,nx+1,nx+2

    @tf.function
    def compute_solution_allTimes(self, nb_t, w_init):
        (b, nx, d) = w_init.shape

        res_w = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx, d], dynamic_size=False, clear_after_read=True)
        res_Fnum = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx+1, d], dynamic_size=False, clear_after_read=True)
        res_L = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx+2, d], dynamic_size=False, clear_after_read=True)

        w = w_init
        for t in tf.range(nb_t):
            w, Fnum, L = self.one_time_step(w)
            res_w = res_w.write(t, w)
            res_Fnum = res_Fnum.write(t, Fnum)
            res_L = res_L.write(t, L)

        return res_w.stack(),res_Fnum.stack(),res_L.stack()


def operator(n,dt,dx,Re):
    coef=-dt/(dx**2 * Re)
    non_diag=tf.ones([n])*coef
    diag=tf.ones([n])*(1-2*coef)
    return tf.linalg.LinearOperatorTridiag([non_diag, diag, non_diag],diagonals_format='sequence').to_dense()



class SolverWithDiffusion(SolverFromL):

    def __init__(self,nx,dx,dt,BC,flux_fn,Re):
        super().__init__(nx,dx,dt,BC,flux_fn)
        self.Re=Re
        self.A=operator(nx,dt,dx,Re)
        self.LU=tf.linalg.lu(self.A)


    def one_time_step_solve(self,w):
        w_star, Fnums, Ls = super().one_time_step(w)
        w=tf.linalg.solve(self.A,w_star)
        return w,Fnums,Ls

    def one_time_step(self,w):
        w_star, Fnums, Ls = super().one_time_step(w)
        w=tf.linalg.lu_solve(*self.LU,w_star)
        return w,Fnums,Ls








def flux_fn_burger(w):
    return w**2 / 2


def test_solvers():
    nx = 400
    BC = "neumann"
    #CFL = 0.5
    xmin = 0.
    xmax = 1.
    tmax=0.1
    dx = (xmax - xmin) / nx
    dt_over_dx = 0.3
    dt = dt_over_dx * dx
    nt = int(tmax / dt)

    #ts = tf.linspace(0., tmax, nt)
    xs = tf.linspace(xmin, xmax, nx)

    w_init=tf.sin(2*np.pi*xs)[tf.newaxis,:,tf.newaxis]

    Re=5
    if Re is not None:
        solver=SolverWithDiffusion(nx,dx,dt,BC,flux_fn_burger,Re)
    else:
        solver=SolverFromL(nx,dx,dt,BC,flux_fn_burger)

    U,_,_ = solver.compute_solution_allTimes(nt,w_init)
    U=U[:,0,:,0]

    fig,ax=plt.subplots()
    grid=Grid_to_plot(nx,xmin,xmax,nt,tmax)
    grid.plot_several_times(ax,U)

    plt.show()


if __name__=="__main__":
    test_solvers()