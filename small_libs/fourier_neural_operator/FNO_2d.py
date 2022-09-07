
pp=print
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(0)

"""
Idée: mettre beaucoup de mode
pénalisation pour filtrer

Permettre un décalage de grille
"""


def reflexive_padding_2d(W: tf.Tensor, pad_1: int,pad_2:int):
    return tf.pad(W,[[0,0],[pad_1,pad_1],[pad_2,pad_2],[0,0]],mode="REFLECT")


class SpectralConv2d(tf.keras.layers.Layer):

    def __init__(self, in_channels:int, out_channels, modes1,modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.weights1  = self.get_complex_weights_2d(in_channels,out_channels)
        self.weights2  = self.get_complex_weights_2d(in_channels,out_channels)

    def get_complex_weights_2d(self,in_channels,out_channels):
        scale = (1 / (in_channels*out_channels))
        real = tf.random.uniform([in_channels, out_channels, self.modes1,self.modes2])
        img  = tf.random.uniform([in_channels, out_channels, self.modes1,self.modes2])
        return tf.Variable(tf.complex(real,img)*scale)

    def channel_mix(self, inputs, weights):
        # (batch, in_channel, nx,ny ), (in_channel, out_channel, nx) -> (batch, out_channel, ny)
        return tf.einsum("bixy,ioxy->boxy", inputs, weights)


    def call(self, A):

        #on met les 2 axes d'espace en dernier, car rfft2d agit sur les 2 derniers axies
        A=tf.transpose(A, [0, 3, 1, 2])

        #Compute Fourier coeffcients
        A_ft = tf.signal.rfft2d(A)  # shape=(...,A.shape[-2],A.shape[-1]//2+1)

        """
        Attention au rfft
        [1,1,n1,10]=> [1,1,n1,6]
        [1,1,n1,9]=>  [1,1,n1,5]
        """

        # Multiply relevant Fourier modes
        out_ft_corner_SO = self.channel_mix(A_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft_corner_NO = self.channel_mix(A_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out_ft_corner_SO =tf.pad(out_ft_corner_SO,[[0,0],[0,0],[0,A.shape[2] - self.modes1], [0,A_ft.shape[3] - self.modes2]])
        out_ft_corner_NO =tf.pad(out_ft_corner_NO,[[0,0],[0,0],[A.shape[2]  - self.modes1,0],[0,A_ft.shape[3] - self.modes2]])
        out_ft=out_ft_corner_SO+out_ft_corner_NO
        #Return to physical space
        A = tf.signal.irfft2d(out_ft, fft_length=[A.shape[2],A.shape[3]]) #fft_length=

        #on remet l'axe des channel en dernier
        A=tf.transpose(A, [0, 2, 3, 1])

        return A



class FNO2d(tf.keras.Model):
    def __init__(self, modes:int, width:int,out_channels:int,pad_prop=0.05):
        super().__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.out_channels=out_channels

        self.pad_prop=pad_prop #on peut mettre à zéro si les inputs et les outputs sont périodiques

        print(f"modèle FNO2d crée avec comme hyperparamètre: modes:{modes}, width:{width}, pad_prop:{self.pad_prop} ")

        self.fc0 = tf.keras.layers.Dense(self.width) # input channel is dim_a+1: (a(A), A)

        self.convs=[SpectralConv2d(self.width,self.width,self.modes1,self.modes2) for _ in range(4)]
        self.ws=[tf.keras.layers.Conv2D(self.width,1) for _ in range(4)]

        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(self.out_channels)


    @tf.function
    def call(self, A):

        pad_1=int(A.shape[1]*self.pad_prop)
        pad_2 = int(A.shape[2] * self.pad_prop)

        A=reflexive_padding_2d(A,pad_1,pad_2)

        A = self.fc0(A)

        for i,(layer,w) in enumerate(zip(self.convs,self.ws)):
            x1=layer.call(A)
            x2=w(A)
            A= x1 + x2
            if i != len(self.convs)-1:
                A=tf.nn.gelu(A)

        A=A[:,pad_1:-pad_1,pad_2:-pad_2,:]

        A = self.fc1(A)
        A = tf.nn.gelu(A)
        A = self.fc2(A)

        return A

def test_spectral_conv_2D():
    """"""
    batch_size = 1
    in_channels = 2
    out_channels = 3
    modes1 = 5
    modes2 = 6

    nx=10 * modes1 + 1 #doit être plus grand que modes
    ny=5 * modes2

    spectral_conv = SpectralConv2d(in_channels, out_channels, modes1,modes2)

    X = tf.ones([batch_size,nx,ny,in_channels])
    Y = spectral_conv.call(X)

    assert Y.shape == (batch_size,nx ,ny,out_channels), f"Y.shape:{Y.shape} must be equal to (batch_size,nx,out_channels):{(batch_size,nx,ny, out_channels)} "

    print("X.shape", X.shape)
    print("Y.shape", Y.shape)


def test_FNO2d():

    batch_size = 7
    in_channels = 1
    width_model = 17
    modes = 5
    nx=10 * modes
    ny=5 * modes
    X = tf.ones([batch_size,nx ,ny, in_channels])
    print("X.shape", X.shape)

    model = FNO2d(modes, width_model)
    Y = model.call(X)

    assert Y.shape == (batch_size,nx, ny ,1), f"Y.shape:{Y.shape} must be equal to (batch_size,nx,1):{(batch_size,nx,ny, 1)} "

    print("Y.shape", Y.shape)




class DataCreator:
    def __init__(self,nx,ny,batch_size):
        self.nx=nx
        self.ny=ny
        self.xx,self.yy=tf.meshgrid(tf.linspace(0.,1,nx),tf.linspace(0.,1,ny)) #(nx,ny)
        self.xx=self.xx[tf.newaxis,:,:,tf.newaxis]
        self.yy=self.yy[tf.newaxis,:,:,tf.newaxis]

        self.batch_size=batch_size
        self.A_fun = lambda x, y , nu1,nu2: tf.sin(x*nu1)*tf.sin(y*nu2)
        self.U_fun = lambda x, y , nu1,nu2: tf.sin(x*nu1)*tf.sin(y*nu2)**5


    def nu1_nu2_intensity(self):
        nu1 = tf.math.floor(tf.random.uniform(minval=1, maxval=5, shape=[self.batch_size, 1, 1, 1]))
        nu2 = tf.math.floor(tf.random.uniform(minval=1, maxval=5, shape=[self.batch_size, 1, 1, 1]))
        intensity = tf.random.uniform(minval=-1, maxval=1, shape=[self.batch_size, 1, 1, 1])
        return nu1,nu2,intensity


    def call(self):
        nu1, nu2, intensity=self.nu1_nu2_intensity()
        A = self.A_fun(self.xx, self.yy,nu1,nu2)*intensity
        U = self.U_fun(self.xx, self.yy,nu1,nu2)*intensity
        return A,U

    def loss(self,U_true,U_pred):
        return tf.reduce_sum((U_true-U_pred)**2)/self.nx/self.ny/self.batch_size


def test_overfit_2d():

    batch_size=32
    matrixCreator_train=DataCreator(30,30,batch_size)
    model = FNO2d(modes=16, width=40)

    optimizer=tf.keras.optimizers.Adam(1e-3)

    losses=[]
    try:
        for i in range(2000):
            A,U=matrixCreator_train.call()

            with tf.GradientTape() as tape:
                U_hat=model.call(A)
                loss=matrixCreator_train.loss(U,U_hat)
            grad=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))

            losses.append(loss.numpy())

            if i%10==0:
                print("MSE train error",loss.numpy())

    except KeyboardInterrupt:
        pass

    fig,ax=plt.subplots()
    ax.plot(losses)
    ax.set_title("train loss")
    ax.set_yscale("log")

    plt.show()

    plot_results(matrixCreator_train,model)

    nb=10
    matrixCrea_test=DataCreator(50,50,nb)
    plot_results(matrixCrea_test,model)




def plot_results(data_creator:DataCreator, model):
    nb=data_creator.batch_size

    fig,axs=plt.subplots(nb,4,sharex="all",sharey="all",figsize=(20,2*nb))
    A,U=data_creator.call()
    U_hat=model.call(A)

    print(f"MSE error for resolution nx:{data_creator.nx}, ny:{data_creator.ny}", data_creator.loss(U, U_hat))

    for i in range(nb):
        error = U[i, :,:, 0] - U_hat[i, :,:, 0]
        axs[i,0].imshow(A[i,:,:,0])
        axs[i, 1].imshow(U[i, :, :, 0])
        axs[i, 2].imshow(U_hat[i, :, :, 0])
        axs[i, 3].imshow(error)

    axs[0, 0].set_title("input")
    axs[0, 1].set_title("output")
    axs[0, 2].set_title("prediction")
    axs[0, 3].set_title("error")

    plt.show()


if __name__ == "__main__":
    #test_spectral_conv_2D()
    #test_FNO2d()
    test_overfit_2d()




