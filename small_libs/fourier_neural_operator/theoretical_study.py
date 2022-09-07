from small_libs.paddings.paddings import pad_nd

from paddings.paddings_right import pad_nd_right

pp=print
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=10000,precision=4)

def case_study():

    case="non_periodic"
    is_right=True

    if case=="non_periodic":
        freq = 5.15
        cst = freq * 2 * np.pi
        fonc = lambda x: tf.sin(cst * x ** 2)
        fonc_prime = lambda x: cst * tf.cos(cst * x ** 2) * 2 * x

    elif case=="non_periodic_smooth":
        nu = 5.5
        fonc = lambda x: tf.cos(nu * 2 * np.pi * x) + tf.sin(2 * nu * 2 * np.pi * x)
        fonc_prime = lambda x: -nu * 2 * np.pi * tf.sin(nu * 2 * np.pi * x) + 2 * nu * 2 * np.pi * tf.cos(
            2 * nu * 2 * np.pi * x)

    else:
        nu = 4
        fonc = lambda x: tf.cos(nu * 2 * np.pi * x) + tf.sin(2 * nu * 2 * np.pi * x)
        fonc_prime = lambda x: -nu * 2 * np.pi * tf.sin(nu * 2 * np.pi * x) + 2 * nu * 2 * np.pi * tf.cos(
            2 * nu * 2 * np.pi * x)

    print("case",case)


    if is_right:
        for kind_padding in ["linear_periodizing_right_padding",None]:
            print("kind_padding:", kind_padding)
            if kind_padding is None:
                pad_prop = 0
            else:
                pad_prop = 0.1
            all_case_right(fonc, fonc_prime, pad_prop, kind_padding)

    else:
        for kind_padding in ["zero_padding","dirichlet_padding","smooth_padding","smooth_periodizing_padding",None]:
            print("kind_padding:",kind_padding)
            if kind_padding is None:
                pad_prop=0
            else:
                pad_prop=0.1
            all_case(fonc,fonc_prime,pad_prop,kind_padding)






def all_case(fonc,fonc_prime,pad_prop,kind_padding):

    mode=20

    fig, axs = plt.subplots(4, 3,figsize=(10,10))
    T=1.


    def one_N(N,T):

        y_fft_len = N//2 if N%2==1 else N//2+1

        x = np.linspace(0., T, N,endpoint=False).astype(np.float32)
        y_init = fonc(x)
        y_prime=fonc_prime(x)

        pad=0
        if pad_prop>0:
            pad=int(N*pad_prop)
            if pad>=N:
                pad=N-1
            y_init=pad_nd(y_init,kind_padding,[pad],[0])
            y_prime=pad_nd(y_prime,kind_padding,[pad],[0])

            N=N+2*pad
            y_fft_len = N // 2 if N % 2 == 1 else N // 2 + 1
            T=T+2*pad_prop*T
            x = np.linspace(-T*pad_prop, T+T*pad_prop, N, endpoint=False).astype(np.float32)


        def recons(y_fft):
            zero = tf.zeros([y_fft_len - mode], dtype=tf.complex64)
            y_fft_pad = tf.concat([y_fft, zero], axis=0)
            y_recons = tf.signal.irfft(y_fft_pad) * N
            return y_recons

        def transform(y):
            y_fft=tf.signal.rfft(y)/N
            y_fft=y_fft[:mode]
            return y_fft,recons(y_fft)

        def unpad(a):
            if pad==0:
                return a
            else:
                return a[pad:-pad]

        y_fft,y_recons=transform(y_init)
        y_fft_prime,y_recons_prime=transform(y_prime)

        k=2*np.pi*tf.cast(tf.range(mode),tf.float32)/T
        k=tf.complex(tf.zeros_like(k),k)
        y_fft_k=y_fft*k
        y_recons_k=recons(y_fft_k)

        mse_recons=tf.reduce_mean(tf.abs(unpad(y_init-y_recons)))
        mse_recons_prime=tf.reduce_mean(tf.abs(unpad(y_prime-y_recons_prime)))
        mse_k=tf.reduce_mean(tf.abs(unpad(y_prime-y_recons_k)))

        print(f"mse_recons:{mse_recons},mse_recons_prime:{mse_recons_prime},mse_k:{mse_k}")

        unpad_before_plot=False
        if unpad_before_plot:
            x=unpad(x)
            y_init=unpad(y_init)
            y_recons=unpad(y_recons)
            y_prime=unpad(y_prime)
            y_recons_prime=unpad(y_recons_prime)
            y_recons_k=unpad(y_recons_k)

        axs[0,0].plot(x, y_init)
        axs[1,0].plot(tf.math.real(y_fft),".")
        axs[2,0].plot(tf.math.imag(y_fft),".")
        axs[3,0].plot(x,y_recons)

        axs[0,1].plot(x, y_prime)
        axs[1, 1].plot(tf.math.real(y_fft_prime), ".")
        axs[2, 1].plot(tf.math.imag(y_fft_prime), ".")
        axs[3, 1].plot(x, y_recons_prime)


        axs[0,2].plot(x, y_prime)
        axs[1, 2].plot(tf.math.real(y_fft_k), ".")
        axs[2, 2].plot(tf.math.imag(y_fft_k), ".")
        axs[3, 2].plot(x, y_recons_k)



    axs[0,0].set_title("y_init")
    axs[1,0].set_title("fft real")
    axs[2,0].set_title("fft imag")
    axs[3,0].set_title("recons")

    axs[0,1].set_title("y_prime")
    axs[1,1].set_title("fft real")
    axs[2,1].set_title("fft imag")
    axs[3,1].set_title("recons")


    axs[0,2].set_title("y_prime")
    axs[1,2].set_title("fft_k real")
    axs[2,2].set_title("fft_k imag")
    axs[3,2].set_title("recons_k")

    fig.suptitle(kind_padding)

    one_N(50,T)
    one_N(100,T)
    one_N(200,T)

    fig.tight_layout()
    plt.show()





def all_case_right(fonc,fonc_prime,pad_prop,kind_padding):

    mode=20

    fig, axs = plt.subplots(4, 2,figsize=(10,10))
    T=1.


    def one_N(N,T):

        y_fft_len = N//2 if N%2==1 else N//2+1

        x = np.linspace(0., T, N,endpoint=False).astype(np.float32)
        y_init = fonc(x)

        pad=0
        if pad_prop>0:
            pad=int(N*pad_prop)
            if pad>=N:
                pad=N-1
            y_init=pad_nd_right(y_init,kind_padding,[pad],[0])

            N=N+pad
            y_fft_len = N // 2 if N % 2 == 1 else N // 2 + 1
            T=T+pad_prop*T
            x = np.linspace(0, T+T*pad_prop, N, endpoint=False).astype(np.float32)


        y_fft = tf.signal.rfft(y_init) / N
        y_fft = y_fft[:mode]

        def recons(y_fft):
            zero = tf.zeros([y_fft_len - mode], dtype=tf.complex64)
            y_fft_pad = tf.concat([y_fft, zero], axis=0)
            y_recons = tf.signal.irfft(y_fft_pad,fft_length=(N,)) * N

            return y_recons
        y_recons=recons(y_fft)


        k=2*np.pi*tf.cast(tf.range(mode),tf.float32)/T
        k=tf.complex(tf.zeros_like(k),k)
        y_fft_k=y_fft*k
        y_recons_k=recons(y_fft_k)


        def unpad(a):
            if pad==0:
                return a
            else:
                return a[:-pad]


        y_prime = fonc_prime(x)
        mse_recons=tf.reduce_mean(tf.abs(unpad(y_init-y_recons)))
        mse_k=tf.reduce_mean(tf.abs(unpad(y_prime-y_recons_k)))

        print(f"mse_recons:{mse_recons},mse_k:{mse_k}")

        unpad_before_plot=False
        if unpad_before_plot:
            x=unpad(x)
            y_init=unpad(y_init)
            y_recons=unpad(y_recons)
            y_prime=unpad(y_prime)
            y_recons_k=unpad(y_recons_k)

        axs[0,0].plot(x, y_init)
        axs[1,0].plot(tf.math.real(y_fft),".")
        axs[2,0].plot(tf.math.imag(y_fft),".")
        axs[3,0].plot(x,y_recons)


        axs[0,1].plot(x, y_prime)
        axs[1, 1].plot(tf.math.real(y_fft_k), ".")
        axs[2, 1].plot(tf.math.imag(y_fft_k), ".")
        axs[3, 1].plot(x, y_recons_k)



    axs[0,0].set_title("y_init")
    axs[1,0].set_title("fft real")
    axs[2,0].set_title("fft imag")
    axs[3,0].set_title("recons")

    axs[0,1].set_title("y_prime")
    axs[1,1].set_title("fft real")
    axs[2,1].set_title("fft imag")
    axs[3,1].set_title("recons via fft")


    fig.suptitle(kind_padding)

    one_N(50,T)
    one_N(100,T)
    one_N(200,T)

    fig.tight_layout()
    plt.show()




def all_case_prefiltering(fonc,fonc_prime):

    mode=15

    filt = tf.cast(tf.range(mode), tf.float32)
    filt /= mode
    #filt = tf.math.sigmoid(10*filt)
    filt = tf.math.pow(filt,0.5)
    #filt = tf.ones_like(filt)
    filt = filt[::-1]

    plt.plot(filt)
    plt.show()

    filt = tf.complex(filt, tf.zeros_like(filt))

    fig, axs = plt.subplots(4, 3,figsize=(10,10))
    T=1.

    def one_N(N,T):

        y_fft_len = N//2 if N%2==1 else N//2+1

        x = np.linspace(0., T, N,endpoint=False).astype(np.float32)
        y_init = fonc(x)
        y_prime=fonc_prime(x)


        def recons(y_fft):
            zero = tf.zeros([y_fft_len - mode], dtype=tf.complex64)
            y_fft_pad = tf.concat([y_fft, zero], axis=0)
            y_recons = tf.signal.irfft(y_fft_pad) * N
            return y_recons



        def transform(y):
            y_fft=tf.signal.rfft(y)/N
            y_fft=y_fft[:mode]*filt

            return y_fft,recons(y_fft)


        y_fft,y_recons=transform(y_init)
        y_fft_prime,y_recons_prime=transform(y_prime)

        k=2*np.pi*tf.cast(tf.range(mode),tf.float32)/T
        k=tf.complex(tf.zeros_like(k),k)
        y_fft_k=y_fft*k
        y_recons_k=recons(y_fft_k)

        mse_recons=tf.reduce_mean(tf.abs((y_init-y_recons)))
        mse_recons_prime=tf.reduce_mean(tf.abs((y_prime-y_recons_prime)))
        mse_k=tf.reduce_mean(tf.abs((y_prime-y_recons_k)))

        print(f"mse_recons:{mse_recons},mse_recons_prime:{mse_recons_prime},mse_k:{mse_k}")

        axs[0,0].plot(x, y_init)
        axs[1,0].plot(tf.math.real(y_fft),".")
        axs[2,0].plot(tf.math.imag(y_fft),".")
        axs[3,0].plot(x,y_recons)

        axs[0,1].plot(x, y_prime)
        axs[1, 1].plot(tf.math.real(y_fft_prime), ".")
        axs[2, 1].plot(tf.math.imag(y_fft_prime), ".")
        axs[3, 1].plot(x, y_recons_prime)


        axs[0,2].plot(x, y_prime)
        axs[1, 2].plot(tf.math.real(y_fft_k), ".")
        axs[2, 2].plot(tf.math.imag(y_fft_k), ".")
        axs[3, 2].plot(x, y_recons_k)


    axs[0,0].set_title("y_init")
    axs[1,0].set_title("fft real")
    axs[2,0].set_title("fft imag")
    axs[3,0].set_title("recons")

    axs[0,1].set_title("y_prime")
    axs[1,1].set_title("fft real")
    axs[2,1].set_title("fft imag")
    axs[3,1].set_title("recons")


    axs[0,2].set_title("y_prime")
    axs[1,2].set_title("fft_k real")
    axs[2,2].set_title("fft_k imag")
    axs[3,2].set_title("recons_k")

    fig.suptitle("prefiltering")

    one_N(50,T)
    one_N(100,T)
    one_N(200,T)

    fig.tight_layout()
    plt.show()





case_study()
#case_study_filtering()



