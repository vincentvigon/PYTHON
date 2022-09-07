# astuce pour avoir la même syntaxe quand on importe small_libs__ de l'extérieur
import derivator as sl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def test_precision():

    L=3.4
    # si freq est un entier: pas besoin de mettre du padding
    freq=5.15
    cst=freq*2*np.pi/L
    fn=lambda x:tf.sin(cst*x**2)
    fn_x=lambda x: cst*tf.cos(cst*x**2)*2*x

    fig,axs=plt.subplots(4,1,figsize=(10,10))

    Ns = [15, 30, 50, 75, 100, 200, 300, 400, 800,1500]

    errors_df_1=[]
    errors_df_2=[]


    for N in Ns:
        x = np.linspace(0., L, N, endpoint=False).astype(np.float32)
        F = fn(x)
        F_x = fn_x(x)
        F_x_df = sl.Dx(F, L / N)
        error_df_1 = tf.reduce_mean(tf.abs(F_x_df - F_x))
        error_df_2 = tf.reduce_mean(tf.square(F_x_df - F_x))

        errors_df_1.append(error_df_1)
        errors_df_2.append(error_df_2)


    def one_pad_prop(pad_prop):
        print("pad_prop",pad_prop)

        def one_N(N:int):
            print(f"test with N={N}")

            x=np.linspace(0.,L,N,endpoint=False).astype(np.float32)
            F=fn(x)
            F_x=fn_x(x)

            F=F[None,None,:,None]
            F_x=F_x[None,None,:,None]

            derivator=sl.Derivator_fft([2],[L],lambda x:x,pad_prop)
            F_x_pred=derivator(F)
            error_1=tf.reduce_mean(tf.abs(F_x_pred-F_x))
            error_2=tf.reduce_mean(tf.square(F_x_pred-F_x))

            return error_1,error_2

        errors_1=[]
        errors_2=[]

        for N in Ns:
            error_1,error_2=one_N(N)
            errors_1.append(error_1)
            errors_2.append(error_2)

        axs[0].plot(Ns,errors_1,"o-",label=f"mae, pad_prop={pad_prop}")
        axs[1].plot(Ns,errors_2,"o-",label=f"mse, pad_prop={pad_prop}")


    for pad_prop in [0,0.1,0.2,0.5,1]:
        one_pad_prop(pad_prop)

    axs[0].plot(Ns,errors_df_1,"o-",label="mae, finite diff")
    axs[1].plot(Ns,errors_df_2,"o-",label="mse, finite diff")

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    axs[0].legend()
    axs[1].legend()

    t=tf.linspace(0.,L,1000)
    F=fn(t)
    Fx=fn_x(t)
    axs[2].plot(t,F,label='fonc')
    axs[3].plot(t,Fx,label='der')
    axs[2].legend()
    axs[3].legend()

    plt.tight_layout()
    plt.show()


def test_1d():
    L=3.4
    # si freq est un entier: pas besoin de mettre du padding
    freq=5.15
    cst=freq*2*np.pi/L
    fn=lambda x:tf.sin(cst*x**2)
    fn_x=lambda x: cst*tf.cos(cst*x**2)*2*x

    N=200
    x = np.linspace(0., L, N, endpoint=False).astype(np.float32)
    F = fn(x)
    F_x = fn_x(x)

    F = F[None, None, :, None]
    F_x = F_x[None, None, :, None]

    derivator = sl.Derivator_fft( [2], [L], lambda x: x, pad_prop=0.2)
    F_x_pred = derivator(F)

    fig,axs=plt.subplots(3,1,sharex="all")
    axs[0].set_title("function")
    axs[0].plot(F[0,0,:,0])
    axs[1].set_title("true derivative")
    axs[1].plot(F_x[0,0,:,0])
    axs[2].set_title("pred derivative")
    axs[2].plot(F_x_pred[0,0,:,0])

    fig.tight_layout()
    plt.show()


def test_2d():
    # si freq est un entier: pas besoin de mettre du padding
    freq=0.3
    cst=freq*2*np.pi
    fn=lambda a0,a1:tf.sin(cst*a0**2)*a1
    fn_a0=lambda a0,a1: tf.cos(cst*a0**2)*(cst*2*a0)*a1
    fn_a1=lambda a0,a1:tf.sin(cst*a0**2)

    fn_a0a0 = lambda a0, a1: tf.cos(cst*a0**2)*(cst*2*a0)**2*a1+tf.sin(cst*a0**2)*(cst*2*a0)*a1  \
                             + tf.cos(cst*a0**2)*(cst*2)*a1
    fn_a1a1 = lambda a0, a1: tf.zeros_like(a1)
    fn_a0a1=lambda a0,a1: tf.cos(cst*a0**2)*(cst*2*a0)

    Ns=[200,300]
    Ls=[2.4,3.4]

    a0 = np.linspace(0., Ls[0], Ns[0]).astype(np.float32)
    a1 = np.linspace(0., Ls[1], Ns[1]).astype(np.float32)
    # Attention, meshgrid est prévu pour des absicisse, ordonnée. Il faut donc l'inverser
    aa0,aa1=tf.meshgrid(a0,a1)
    aa0, aa1=tf.transpose(aa0),tf.transpose(aa1)
    print("aa",aa0.shape,aa1.shape)

    F = fn(aa0,aa1)
    DF={}
    for f,name in zip([fn_a0,fn_a1,fn_a0a0,fn_a0a1,fn_a1a1],["a0","a1","a0^2","a0a1","a1^2"]):
        DF[name]=f(aa0,aa1)

    derivator = sl.Derivator_fft([0, 1], Ls,
        formula=lambda a0, a1: {"a0": a0, "a1": a1, "a0^2": a0 ** 2, "a1^2": a1 ** 2,"a0a1":a0*a1})
    DF_fft=derivator(F)

    extent = [0., Ls[1], 0., Ls[0]]
    fig, axs = plt.subplots(6,2, figsize=(5, 10),sharex="all")
    axs[0, 0].imshow(F, extent=extent, cmap="jet", origin="lower")
    axs[0, 0].set_title("initial function")

    vmin,vmax=-4,4
    for i,key in enumerate(["a0","a1","a0^2","a0a1","a1^2"]):
        axs[i+1,0].imshow(DF[key],extent=extent,cmap="jet",origin="lower",vmin=vmin,vmax=vmax)
        axs[i + 1, 0].set_title(key+"_true")
        axs[i+1,1].imshow(DF_fft[key],extent=extent,cmap="jet",origin="lower",vmin=vmin,vmax=vmax)
        axs[i + 1, 1].set_title(key+"_pred")

    plt.tight_layout()
    plt.show()



