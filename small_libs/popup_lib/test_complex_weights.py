from typing import List, Dict

from fourier_neural_operator.FNO_1d_plus import FNO1d_plus
from popup_lib.popup2 import Abstract_Agent

pp=print
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=10000,precision=4)
import popup2 as pop

class DataCreator1D:
    def __init__(self,nx,batch_size):
        self.nx=nx
        self.batch_size=batch_size

        self.A_fun = lambda x, nu: -tf.sin(nu * x)+0.5*tf.sin(2*nu * x)
        self.U_fun = lambda x, nu: tf.cos(nu * x) ** 5 +0.3*tf.cos(2*nu * x)

        self.grid = tf.linspace(0., 1, self.nx)[tf.newaxis, :, tf.newaxis]

    def nus_intensities(self):
        nus=tf.floor(tf.random.uniform(minval=2, maxval=30, shape=[self.batch_size, 1, 1]))
        intensities=tf.random.uniform(minval=0.1,maxval=1.,shape=[self.batch_size,1,1])
        return nus,intensities

    def call(self):
        nus, intensities = self.nus_intensities()
        A = self.A_fun(self.grid, nus) * intensities
        U = self.U_fun(self.grid, nus) * intensities
        return A,U

    def loss(self,U,U_hat):
        return tf.reduce_sum((U - U_hat) ** 2) / self.nx / self.batch_size



class Agent(Abstract_Agent):

    def __init__(self,loss_fn):
        self.loss_fn=loss_fn
        self.modes=16
        self.model = FNO1d_plus(modes=self.modes, width=10, out_channels=1, first_channel_unchanged=True, pad_prop=0.05,freq_mix_size=5)
        self.optimizer = tf.keras.optimizers.Adam()

        self.famparams={}

    def load_XY(self,X,Y):
        self.X=X
        self.Y=Y

    def optimize_and_return_score(self) -> float:
        try:
            for i in range(30):

                with tf.GradientTape() as tape:
                    Y_hat = self.model.call(self.X)
                    loss = self.loss_fn(self.Y, Y_hat)

                grad = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        except KeyboardInterrupt:
            pass

        return -loss.numpy()


    def get_famparams(self) -> Dict[str, float]:
        return self.famparams

    def set_and_perturb_famparams(self, famparam, period_count: int) -> None:
        self.famparams=famparam




    def set_weights(self, weights: List):
        self.model.set_weights(weights)

    def get_copy_of_weights(self) -> List:
        return self.model.get_weights()



def test_overfit():

    data_creator_train=DataCreator1D(nx=50,batch_size=64)
    loss_fn=data_creator_train.loss
    agents=[Agent(loss_fn) for _ in range(4)]

    family_trainer = pop.Family_trainer(agents, period_duration="4 steps", nb_strong=1)


    try:
        for _ in range(10):
            X,Y=data_creator_train.call()
            for agent in agents:
                agent.load_XY(X,Y)
            family_trainer.period()
    except KeyboardInterrupt:
        family_trainer.plot()
        family_trainer.interupt_period()

    family_trainer.plot()

    best_agent=family_trainer.get_best_agent()


    display_results(data_creator_train,best_agent.model)
    data_creator_test=DataCreator1D(100,64)
    display_results(data_creator_test,best_agent.model)

    #model.convs[0].inspect_weiths()




def display_results(data_creator:DataCreator1D,model):

    nb=8# <= data_creator.batch_size
    fig,axs=plt.subplots(nb,4,sharey="row",sharex="all",figsize=(20,2*nb))
    A,U=data_creator.call()
    U_hat = model.call(A)
    print(f"MSE error with resolution nx:{data_creator.nx}:",data_creator.loss(U,U_hat))
    for i in range(nb):
        error = U[i, :, 0] - U_hat[i, :, 0]
        axs[i,0].plot(A[i,:,0])
        axs[i,1].plot(U[i,:,0])
        axs[i,2].plot(U_hat[i,:,0])
        axs[i,3].plot(error)

    axs[0,0].set_title("input")
    axs[0,1].set_title("output")
    axs[0,2].set_title("prediction")
    axs[0,3].set_title("error")

    plt.show()


if __name__ == "__main__":
    test_overfit()