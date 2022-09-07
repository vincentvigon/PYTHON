
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
pp=print



def pad_1d_right(W,kind,pad_size,axis):
    try:
        pad_func=kind_dict[kind]
    except KeyError:
        raise Exception(f"the kind:{kind} is not allowed. Allowed kinds are {kind_dict.keys()}")

    dim = len(W.shape)

    if not (axis==-1 or axis==dim-1):
        permutation = np.arange(dim)
        permutation[-1] = axis%dim
        permutation[axis] = dim - 1

        W=tf.transpose(W,permutation)
        W=pad_func(W,pad_size)
        W=tf.transpose(W,permutation)
    else:
        W=pad_func(W,pad_size)

    return W


def linear_periodizing_right_padding(W: tf.Tensor, pad_size):
    # W.shape: (a,b,c)
    if isinstance(pad_size,tuple) or isinstance(pad_size,list):
        raise Exception("pad_size must be an int")
    else:
        pad_right=pad_size

    left_value = W[..., 0:1] #(a,b,1)
    right_value = W[..., -1:]#(a,b,1)
    amplitude=right_value-left_value#(a,b,1)

    right=tf.linspace(0.,1,pad_right)[::-1] # (p)
    right*=amplitude #(a,b,p)
    right+=left_value
    res=tf.concat([W, right], axis=-1)
    return res


kind_dict={"linear_periodizing_right_padding":linear_periodizing_right_padding}


def test_pad_1d_right():
    x = tf.cast(np.linspace(0., 1, 110, endpoint=False), tf.float32)
    y = tf.sin(x * 2 * np.pi) + (2 * x + 1)
    W = tf.ones([1,1,1])*y[None,:,None] #axis=1

    pad_size = 60

    for kind in kind_dict.keys():
        W_=pad_1d_right(W,kind,pad_size,axis=1)
        plt.plot(W_[0,:,0],label=kind)

    plt.legend(bbox_to_anchor=(0, 0.9),loc='center left')
    plt.show()



def pad_nd_right(W,kind,pad_sizes,axes):
    assert  isinstance(kind,str), "kind must be a string in the following list:"+str(kind_dict.keys())
    assert isinstance(axes,tuple) or isinstance(axes,list), "axes must be a list or a tupe"
    assert isinstance(pad_sizes,tuple) or isinstance(pad_sizes,list), "pad_sizes must be a list or a tupe"

    assert len(pad_sizes)==len(axes)

    for pad_size,axis in zip(pad_sizes,axes):
        W=pad_1d_right(W,kind,pad_size,axis)
    return W



def test_pad_nd_right():
    xmin,xmax=0,2
    ymin,ymax=0,1
    x=tf.linspace(0.,2,40)
    y=tf.linspace(0.,1,40)
    xx,yy=tf.meshgrid(x,y)
    z=tf.sin(10*xx)+yy

    keys=kind_dict.keys()
    nb=len(keys)
    def one(reverse):
        fig,axs=plt.subplots(nb,1,figsize=(5,3*nb))
        if nb==1:
            axs=[axs]
        axes=[1,0] if reverse else [0,1]
        for i, key in enumerate(keys):
            W=pad_nd_right(z,key,[20,20],axes=axes)
            axs[i].imshow(W,cmap="jet",origin="lower",extent=[xmin,xmax,ymin,ymax])
            axs[i].set_title(key)
        plt.tight_layout()

    one(False)
    one(True)

    plt.show()


if __name__=="__main__":

    #test_pad_1d_right()
    test_pad_nd_right()
