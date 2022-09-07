import tensorflow as tf
import matplotlib.pyplot as plt
pp=print
import time


@tf.function
def make_consecutive_windows(Z,window_size):
    print(f"traçage de 'make_consecutive_windows'")
    def make_windows(Z, window_size):
        batch_size, nb_t = Z.shape[0], Z.shape[1]
        nb_windows = nb_t // window_size
        Z = Z[:, :window_size * nb_windows]
        Z = tf.reshape(Z, (batch_size, nb_windows, window_size) + Z.shape[2:])
        return Z

    res=[]
    nb_t=Z.shape[1]
    end_shape=Z.shape[2:]

    Z=Z[:,:nb_t // window_size * window_size - 1] #on racoursi un peu pour que tous les découpages aient le même nombre de tranche
    for i in range(window_size): #attention, on s'arrête avant window_size
        res.append(make_windows(Z[:,i:],window_size))
    res=tf.stack(res,axis=3) #

    bs,nb_windows=res.shape[0],res.shape[1]

    res=tf.reshape(res, (bs, nb_windows * window_size,window_size) + end_shape)
    return res


def test_consecutive_windows_perf():
    """
    On teste avec une entrée de dimension 2.
    Le résultat doit être un dessin dégradé dans le sens de la diagonale.
    """

    b, nb_t=1,200
    window_size=100

    T=tf.cast(tf.range(nb_t),tf.float32)
    X=tf.ones([b,nb_t])

    X=X*T[None,:]
    #traçage
    make_consecutive_windows(X, window_size)
    ti0=time.time()
    X_w=None
    for _ in range(100):
        X_w=make_consecutive_windows(X,window_size) #b,nb_window*window_size (all shift) ...
    duration=time.time()-ti0
    print("duration",duration)
    assert duration<0.1
    """
    Sans @tf.function: 3.5 secondes !!!
    """

    print("X_w",X_w.shape)
    plt.matshow(X_w[0,:,:])
    plt.show()





def test_consecutive_windows():
    """
    On teste avec des entrées de dimension 4 puis 3
    Le résultat doit être
    [[ 0.  1.  2.  3.  4.]
     [ 1.  2.  3.  4.  5.]
     [ 2.  3.  4.  5.  6.]
     [ 3.  4.  5.  6.  7.]
     [ 4.  5.  6.  7.  8.]
     [ 5.  6.  7.  8.  9.]
     [ 6.  7.  8.  9. 10.]
     [ 7.  8.  9. 10. 11.]
     [ 8.  9. 10. 11. 12.]
     [ 9. 10. 11. 12. 13.]
     [10. 11. 12. 13. 14.]
     [11. 12. 13. 14. 15.]
     [12. 13. 14. 15. 16.]
     [13. 14. 15. 16. 17.]
     [14. 15. 16. 17. 18.]]
    """
    b, nb_t, nb_part, dimX=1,20,1,1
    window_size=5

    T=tf.cast(tf.range(nb_t),tf.float32)
    X=tf.ones([b,nb_t , nb_part, dimX])

    X=X*T[None,:,None,None]
    print("X",X[0,:,0,0])

    X_w=make_consecutive_windows(X,window_size) #b,nb_window*window_size (all shift),window_size,nb_part,dimX
    print("X_w",X_w[0,:,:,0,0])

    dimY=1
    Y=tf.ones([b,nb_t,dimY])
    Y=Y*T[None,:,None]
    Y_w=make_consecutive_windows(Y,window_size)

    print("Y_w",Y_w[0,:,:,0])


if __name__=="__main__":
    test_consecutive_windows()
    test_consecutive_windows_perf()
