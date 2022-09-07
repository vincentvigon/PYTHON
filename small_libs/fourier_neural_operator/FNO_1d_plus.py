from small_libs.paddings.paddings import pad_nd
from paddings.paddings_right import pad_nd_right
pp=print
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SpectralConv1d_plus(tf.keras.layers.Layer):

    def __init__(self,
            in_channels:int,
            out_channels:int,
            modes:int,
            first_channel_unchanged:bool,
            N_out_imposed:int or None,
            freq_mix_size:int,
            channel_last: bool
    ):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.N_out_imposed=N_out_imposed
        self.first_channel_unchanged=first_channel_unchanged
        self.freq_mix_size= freq_mix_size
        self.channel_last=channel_last

        self.multiplicative_weights =tf.Variable(self.get_complex_weights(),name="mutliplicative")


        self.kerner_for_freq_conv_real=None
        self.kerner_for_freq_conv_imag=None
        if self.freq_mix_size>1:
            self.kerner_for_freq_conv_real=tf.Variable(self.get_kernel_for_freq_conv())
            self.kerner_for_freq_conv_imag=tf.Variable(self.get_kernel_for_freq_conv())

    def get_complex_weights(self):
        if self.first_channel_unchanged:
            return self.get_complex_weights_for_first_channel_unchanged()
        else:
            return self.get_complex_weights_classic()


    def inspect_weiths(self):

        def plot_conv_kernels(kernel1,kernel2,title):
            _, in_ch, out_ch = kernel1.shape
            fig, axs = plt.subplots(in_ch,out_ch,figsize=(20,20),sharey="all",sharex="all")
            fig.suptitle(title)

            for i in range(in_ch):
                for j in range(in_ch):
                    axs[i,j].plot(kernel1[:,i,j])
                    axs[i,j].plot(kernel2[:,i,j])
            plt.show()

        def plot_mult_kernels(kernel1,kernel2,title):
            in_ch, out_ch,_ = kernel1.shape
            fig, axs = plt.subplots(in_ch,out_ch,figsize=(20,20),sharey="all",sharex="all")
            fig.suptitle(title)

            for i in range(in_ch):
                for j in range(in_ch):
                    axs[i,j].plot(kernel1[i,j,:])
                    axs[i,j].plot(kernel2[i,j,:])
            plt.show()


        if self.kerner_for_freq_conv_real is not None:
            plot_conv_kernels(self.kerner_for_freq_conv_real,self.kerner_for_freq_conv_imag,"kerner_for_freq_conv real/imag")

        plot_mult_kernels(tf.math.real(self.multiplicative_weights),tf.math.imag(self.multiplicative_weights),"multiplicative filter real/imag")


    def get_complex_weights_classic(self):
        scale = (1 / (self.in_channels*self.out_channels))
        real = tf.random.uniform([self.in_channels, self.out_channels, self.modes])
        img =  tf.random.uniform([self.in_channels, self.out_channels, self.modes])
        return tf.complex(real,img)*scale

    def get_complex_weights_for_first_channel_unchanged(self):
        scale = (1 / (self.in_channels*self.out_channels))
        real = np.random.uniform(0.,1,size=[self.in_channels, self.out_channels, self.modes]).astype(np.float32)
        img =  np.random.uniform(0.,1,size=[self.in_channels, self.out_channels, self.modes]).astype(np.float32)
        real[:,0,:]=1
        img[:,0,:]=0.
        return tf.complex(real,img)*scale


    def channel_mix(self, inputs, weights):
        # (batch, in_channel, mode ), (in_channel, out_channel, mode) -> (batch, out_channel, mode)
        """ sum_i inputs[b,i,A] weights[i,o,A]  """
        return tf.einsum("bix,iox->box", inputs, weights)

    def freq_conv(self,B,kernel):
        list_GPU=tf.config.list_physical_devices('GPU')
        if len(list_GPU)==0:
            print("pas de GPU")
            B=tf.transpose(B,[0,2,1])
            B=tf.nn.conv1d(B, kernel, stride=1, padding="VALID", data_format="NWC")
            return tf.transpose(B,[0,2,1])
        else:
            #print("GPU")
            return tf.nn.conv1d(B,kernel,stride=1,padding="VALID",data_format="NCW")


    def get_kernel_for_freq_conv(self):
        # choix identité
        #kernel=np.zeros([self.freq_mix_size ,self.out_channels,self.out_channels]).astype(np.float32)
        #kernel[0,:,:]=1.

        # autre choix
        #kernel = np.ones([self.freq_mix_size * self.out_channels ** 2]).astype(np.float32)/(self.freq_mix_size * self.out_channels ** 2)
        #kernel = np.reshape(kernel, [self.freq_mix_size, self.out_channels, self.out_channels])

        # autre choix
        kernel = tf.random.uniform(minval=-1,maxval=1,shape=[self.freq_mix_size, self.out_channels, self.out_channels])/(self.freq_mix_size * self.out_channels ** 2)
        return kernel



    def do_freq_mix(self,A):
        A_real=tf.math.real(A)
        A_imag=tf.math.imag(A)

        A_real=self.freq_conv(A_real,self.kerner_for_freq_conv_real)
        A_imag=self.freq_conv(A_imag,self.kerner_for_freq_conv_imag)

        return tf.complex(A_real,A_imag)


    @tf.function
    def call(self, x):

        batch_size,N,c=x.shape
        if self.N_out_imposed is None:
            N_out=N
        else:
            N_out=self.N_out_imposed

        if self.channel_last:
            #on passe en format channel-first, à cause de la fft qui est implémentée ainsi
            x=tf.transpose(x,[0,2,1])
        x_ft = tf.signal.rfft(x)/N

        # Multiply relevant Fourier modes
        assert N>=2*self.modes,f"la taille du signal en entrée={N} doit être deux fois plus grande que le nombre de mode={self.modes}"

        out_ft = self.channel_mix(x_ft[:, :, :self.modes], self.multiplicative_weights)

        if self.freq_mix_size>1:
            out_ft=self.do_freq_mix(out_ft) #diminue un peu la taille
        """
        Attention au rfft
        le nombre d'élément de rfft(x) c'est N//2+1
        [...,10]=> [...,6]
        [...,9]=>  [...,5]
        """
        to_add=N_out//2+1 - out_ft.shape[2]
        tf.assert_greater(to_add,0,f"la taille du signal en sortie est trop petite pour le nombre de mode demandé")
        out_ft_pad=tf.pad(out_ft, [[0, 0], [0, 0], [0, to_add]])

        #Return to physical space
        x = tf.signal.irfft(out_ft_pad,fft_length=[N_out])*N_out

        if self.channel_last:
            x=tf.transpose(x,[0,2,1])

        return x


class FNO1d_plus(tf.keras.Model):

    def __init__(self, modes:int, width:int,out_channels:int,nb_layer=4,first_channel_unchanged=False,freq_mix_size=0,pad_prop=0.1,pad_kind="zero_padding",verbose=False):
        """
        @param modes:int, number of fourier coef keept
        @param width:int, nb_channels of hidden layers
        @param out_channels:int, dimension of output
        @param nb_layer:int,
        @param first_channel_unchanged:bool, if True, at the beginning of the training, the first channel of each hidden layer is simply the spectrum
        @param freq_mix_size:int, size of the convolutionnal kernel applyiend the frequency layers
        @param pad_prop:float, the padding is int(pad_prof*signal_size) at each border
        @param pad_kind:string, kind of padding. Eg: neumann_padding, smooth_padding,zero_padding
        @param verbose:bool, if True, more message are printed
        """
        super().__init__()

        self.modes = modes
        self.width = width
        self.out_channels=out_channels
        self.first_channel_unchanged=first_channel_unchanged
        self.pad_prop=pad_prop
        self.freq_mix_size=freq_mix_size
        self.pad_kind=pad_kind
        self.verbose=verbose


        if verbose:
            print(f"modèle FNO1d crée avec comme hyperparamètre: modes:{modes}, width:{width},nb layers:{nb_layer}, pad_prop:{pad_prop} ")


        self.fc0 = tf.keras.layers.Dense(self.width)

        self.convs=[SpectralConv1d_plus(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            first_channel_unchanged=self.first_channel_unchanged,
            N_out_imposed=None ,
            freq_mix_size=self.freq_mix_size,
            channel_last=True #todo: essayer de faire la multiplication matricielle en channel_firt pour éviter les transpositions, tester
        ) for _ in range(nb_layer)]

        self.ws=[tf.keras.layers.Conv1D(self.width,1) for _ in range(nb_layer)]
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(self.out_channels)


    @tf.function
    def call(self, A):

        pad=int(A.shape[1]*self.pad_prop)

        if self.pad_kind is None or self.pad_kind == "no_padding":
            pad = 0

        if self.verbose:
            print(f"traçage de la méthode call de FNO1d_plus, pad={pad}")


        #A=reflexive_padding(A,pad)
        if pad>0:

            if self.pad_kind=="linear_periodizing_right_padding":
                A=pad_nd_right(A,self.pad_kind,[pad], axes=[1])
            else:
                A=pad_nd(A, self.pad_kind, [pad], axes=[1])

        A = self.fc0(A)

        for i,(layer,w) in enumerate(zip(self.convs,self.ws)):
            x1=layer.call(A)
            x2=w(A)
            A= x1 + x2
            if i!=len(self.convs)-1:
                A=tf.nn.gelu(A)

        if pad>0:
            if self.pad_kind == "linear_periodizing_right_padding":
                A=A[:,:-pad,:]
            else:
                A=A[:,pad:-pad,:]

        A = self.fc1(A)
        A = tf.nn.gelu(A)
        A = self.fc2(A)

        return A



def test_FNO1d():

    batch_size = 2
    in_channels = 3
    out_channels = 4
    N_in=50
    modes=5
    width_model=8

    X = tf.ones([batch_size,N_in ,in_channels])*tf.range(0.,N_in)[tf.newaxis,:,tf.newaxis]

    model = FNO1d_plus(modes, width_model,out_channels,first_channel_unchanged=True,pad_prop=0.1,freq_mix_size=0,pad_kind="smooth_periodizing_padding")
    Y = model.call(X)

    assert Y.shape == (batch_size, N_in ,out_channels), f"Y.shape:{Y.shape} must be equal to (batch_size,N_out,out_channels):{(batch_size,N_in, out_channels)} "

    print("Y.shape", Y.shape)

# class DataCreator1D:
#     def __init__(self,nx,batch_size):
#         self.nx=nx
#         self.batch_size=batch_size
#
#         self.X_fun  = lambda x,nu: tf.sin(nu * 2 * np.pi * x ** 2)
#         self.Y_fun = lambda x,nu: nu * tf.cos(nu * 2 * np.pi * x ** 2) * 2 * x
#         #
#         # self.A_fun = lambda x, nu: -tf.sin(nu * x)+0.5*tf.sin(2*nu * x)
#         # self.U_fun = lambda x, nu: tf.cos(nu * x) ** 5 +0.3*tf.cos(2*nu * x)
#
#         self.grid = tf.linspace(0., 1, self.nx)[tf.newaxis, :, tf.newaxis]
#
#     def nus_intensities(self):
#         nus=tf.floor(tf.random.uniform(minval=0.5, maxval=5, shape=[self.batch_size, 1, 1]))
#         intensities=tf.random.uniform(minval=0.1,maxval=1.,shape=[self.batch_size,1,1])
#         return nus,intensities
#
#     def call(self):
#         nus, intensities = self.nus_intensities()
#         X = self.X_fun(self.grid, nus) * intensities
#         Y = self.Y_fun(self.grid, nus) * intensities
#         return X,Y
#
#     def loss(self,Y_true,Y_pred):
#         return tf.reduce_sum((Y_true - Y_pred) ** 2) / self.nx / self.batch_size



if __name__ == "__main__":
    test_FNO1d()



