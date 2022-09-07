import fft_nd as sl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time




def test_fft_perf():
    W = tf.random.uniform([20,10,20, 30, 40])
    W = tf.complex(W, W)

    def one(axes_grouping):
        ti0=time.time()
        F=sl.Fft_nd([1,2,3,4],axes_grouping)(W)
        duration=time.time()-ti0
        print("axes_grouping",axes_grouping,"duration:",duration)
        return F

    Fs=[]
    for grouping in [(1,1,1,1),(2,2),(1,2,1),(3,1),(1,1,2)]:
        F=one(axes_grouping=grouping)
        Fs.append(F)


    def mse_fn(A):
        return tf.reduce_mean(tf.square(tf.abs(A)))

    F0=Fs[0]
    for F in Fs[1:]:
        mse=mse_fn(F-F0)
        print(mse)
        assert mse<1e-6

