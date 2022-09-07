import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from small_libs.derivator_fft.fft_nd import Fft_nd
from small_libs.paddings.paddings import pad_nd, unpad_nd, smooth_padding,padding_kind

pp=print



class Derivator_fft:

    def __init__(self,
            input_shape,
            input_dtype,
            axes,
            interval_lenghts,
            formula,
            pad_prop=0.1,
            suppress_padding_on_output=True,
            graph_acceleration=False,
            verbose=False,
            padding_kind=padding_kind.smooth_periodizing_padding
    ):

        self.input_shape=input_shape
        self.input_dtype=input_dtype
        self.axes=axes
        self.interval_lenghts=interval_lenghts
        self.formula=formula
        self.pad_prop=pad_prop
        self.suppress_padding_on_output=suppress_padding_on_output
        self.graph_acceleration=graph_acceleration
        self.verbose=verbose
        self.padding_kind=padding_kind

        self.init_freq_multiplicators()



    def init_freq_multiplicators(self):


        sizes = [self.input_shape[i] for i in self.axes]

        if self.input_dtype==tf.float32 or self.input_dtype==tf.complex64:
            is_32bit=True
        elif self.input_dtype ==tf.float64 or self.input_dtype ==tf.complex128:
            is_32bit=False
        else:
            raise Exception("the tensor given must have dtype in [tf.float32,tf.complex64,tf.float64,tf.complex128]")


        if is_32bit:
            self.dtype_real = np.float32  # tensorflow accepte les dtype numpy
            self.dtype_complex = np.complex64
        else:
            self.dtype_real = np.float64
            self.dtype_complex = np.complex128


        sizes_with_pad = []
        interval_length_with_pad = []

        pads = [int(size * self.pad_prop) for size in sizes]


        for i in range(len(self.axes)):
            if pads[i] >= sizes[i]:
                pads[i] = sizes[i] - 1
            sizes_with_pad.append(sizes[i] + 2 * pads[i])
            interval_length_with_pad.append(self.interval_lenghts[i] * (1 + 2 * self.pad_prop))

        self.pads = pads

        ks = []

        """
        Calcul des vecteurs k, qui, une fois multiplié à la transformée de fourier, produirons la dérivation 
        """
        for i in range(len(self.axes)):
            k_max = sizes_with_pad[i] // 2
            if sizes_with_pad[i] % 2 == 0:
                k = tf.concat([tf.range(0., k_max, dtype=self.dtype_real), tf.range(-k_max, 0., dtype=self.dtype_real)],
                    axis=0)
            else:
                k = tf.concat(
                    [tf.range(0., k_max, dtype=self.dtype_real), [0.], tf.range(-k_max, 0., dtype=self.dtype_real)],
                    axis=0)

            k *= 2. * np.pi / interval_length_with_pad[i]
            # À cause de la multiplication par sqrt(-1), k passe en second argument ci-dessous
            k = tf.complex(tf.zeros_like(k),k)

            """
            Maintenant, il faut étaler ce vecteur dans le bon axe. Par exemple si current_axis=3
            la shape de k sera (1,1,1,size,1,1,...)
            """
            s = [1] * self.axes[i]
            s += [sizes_with_pad[i]]
            s += [1] * (len(self.input_shape) - self.axes[i]-1)

            k = tf.reshape(k, s)
            if self.verbose:
                print(i,"shape ieme tenseur:",s)
            ks.append(k)

        self.freq_factor=self.formula(*ks)

        self.result_type = None
        if isinstance(self.freq_factor,dict):
            self.user_keys=self.freq_factor.keys()
            self.freq_factor=self.freq_factor.values()
            self.result_type = "dict"
        elif isinstance(self.freq_factor,list):
            self.result_type="list"
        elif isinstance(self.freq_factor, tuple):
            self.result_type = "tuple"
        else:
            self.result_type = "scalar"
            self.freq_factor =[self.freq_factor]

        self.fft=Fft_nd(self.axes)
        self.ifft=Fft_nd(self.axes,inverse=True)


    def __call__(self,U):

        if self.graph_acceleration:
            print("traçage de la méthode de dérivation du Derivator_fft",end="")
            ti0=time.time()
            tf_function_obj = tf.function(self.D_initial)
            concrete_function = tf_function_obj.get_concrete_function(U)
            duration=time.time()-ti0
            print(", temps de traçage:",duration)
            return concrete_function(U)
        else:
            return self.D_initial(U)


    #
    # @tf.function
    # def D_graph(self,U):
    #     print("traçage de la méthode D_graph de la classe Derivator_fft")
    #     return self.D_initial(U)


    def D_initial(self, U):
        if isinstance(U, np.ndarray):
            U = tf.constant(U)

        U = pad_nd(U, self.padding_kind, self.pads, axes=self.axes)
        U = tf.cast(U,self.dtype_complex)

        U_ = self.fft(U)

        res=[]
        for freq_factor in self.freq_factor:
            U_der=U_*freq_factor
            U_der=self.ifft(U_der)

            if self.suppress_padding_on_output:
                U_der=unpad_nd(U_der,self.pads,self.axes)

            U_der = tf.cast(U_der, self.input_dtype)
            res.append(U_der)


        if self.result_type=="dict":
            res_dict={}
            for key,val in zip(self.user_keys,res):
                res_dict[key]=val
            res=res_dict
        elif self.result_type=="scalar":
            res=res[0]
        elif self.result_type=="list":
            pass
        elif self.result_type=="tuple":
            res=tuple(res)
        return res



def Dx(W,h,keep_size=True,centred=True,axis=-1):

    permut=None
    if axis!=-1:
        dim=len(W.shape)
        permut = np.arange(dim)
        permut[-1] = axis
        permut[axis] = dim - 1
        W=tf.transpose(W,permut)

    if centred:
        Wx=(W[...,2:]-W[...,:-2])/(2*h)
        if keep_size:
            Wx=smooth_padding(Wx,1)
    else:
        Wx=(W[...,1:]-W[...,:-1])/h
        if keep_size:
            Wx=smooth_padding(Wx,[1,0])

    if axis != -1:
        Wx = tf.transpose(Wx, permut)

    return Wx




def simple_test():
    T0=1
    T1=2
    T2=1.5
    a0=tf.linspace(0.,T0,50)[:,None,None]
    a1=tf.linspace(0.,T1,30)[None,:,None]
    a2=tf.linspace(0.,T2,20)[None,None,:]

    U=a0*a1**2*a2
    print("U",U.shape)

    derivator=Derivator_fft(U.shape,U.dtype,[2,1],[T0,T1],lambda a0,a1:[a0,a1],pad_prop=0,suppress_padding_on_output=False)
    U_a0,U_a1=derivator(U)

    print(U_a0.shape)

    fig,axs=plt.subplots(2,1)
    axs[0].imshow(U_a0[:,:,0])
    axs[1].imshow(U_a1[:,:,0])
    plt.show()



def test_divergence():
    N=10
    a0=tf.linspace(0,1,N)
    a1=tf.linspace(0,1,N)
    a2=tf.linspace(0,1,N)
    a3=tf.linspace(0,1,N)

    a0_=a0[:,None,None,None]
    a1_=a1[None,:,None,None]
    a2_=a2[None,None,:,None]
    a3_=a3[None,None,None,:]

    func=lambda a0,a1,a2,a3:a0**2+a1**2+a2**2+a3**2
    div_func=lambda a0,a1,a2,a3: 2*a0+2*a1+2*a2+2*a3

    F=func(a0_,a1_,a2_,a3_)
    div_F=div_func(a0_,a1_,a2_,a3_)

    derivator_fft=Derivator_fft(F.shape,F.dtype,[0,1,2,3],[1,1,1,1],lambda a0,a1,a2,a3:a0+a1+a2+a3,graph_acceleration=True)

    for i in range(4):
        ti0=time.time()
        div_F_pred=derivator_fft(F)
        duration=time.time()-ti0
        print(f"temps d'execution {i}:",duration)

    print("SANS TRAçAGE")
    derivator_fft=Derivator_fft(F.shape,F.dtype,[0,1,2,3],[1,1,1,1],lambda a0,a1,a2,a3:a0+a1+a2+a3,graph_acceleration=False)

    for i in range(4):
        ti0=time.time()
        div_F_pred=derivator_fft(F)
        duration=time.time()-ti0
        print(f"temps d'execution {i}:",duration)







if __name__=="__main__":
    #test_1d()
    #test_2d()
    simple_test()
    #test_precision()
    #test_divergence()







