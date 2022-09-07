
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
pp=print


class Padding_kind:

    def __init__(self):

        self.zero_padding = "zero_padding"
        self.dirichlet_padding = "dirichlet_padding"
        self.neumann_padding = "neumann_padding"
        self.smooth_padding = "smooth_padding"
        self.periodic_padding = "periodic_padding"
        self.smooth_periodizing_padding = "smooth_periodizing_padding"
        self.linear_periodizing_padding = "linear_periodizing_padding"

        self.key_to_func = {
            self.zero_padding: zero_padding,
            self.dirichlet_padding: dirichlet_padding,
            self.neumann_padding: neumann_padding,
            self.smooth_padding: smooth_padding,
            self.periodic_padding: periodic_padding,
            self.smooth_periodizing_padding: smooth_periodizing_padding,
            self.linear_periodizing_padding: linear_periodizing_padding
        }


    def get_one_function(self, kind):
        return self.key_to_func[kind]

    def get_some_function(self, kinds):
        return [self.key_to_func[kind] for kind in kinds]

    def get_all_function(self):
        return [self.key_to_func[kind] for kind in self.key_to_func.values()]


def tend_to_zero_right(U):
    N=U.shape[-1]
    t=linspace(0.,2,N,U.dtype)
    t=tf.cast(t,U.dtype)
    decrease=tf.exp(-t**4)
    return U*decrease
def tend_to_zero_left(U):
    N=U.shape[-1]
    t=linspace(-2,0,N,U.dtype)
    t=tf.cast(t,U.dtype)
    decrease=tf.exp(-t**4)
    return U*decrease
def test_tend_to_zero():
    U=tf.sin(tf.linspace(0.,15,100))
    U_right=tend_to_zero_right(U)
    U_left=tend_to_zero_left(U)

    fig,axs=plt.subplots(2,1)
    axs[0].plot(U)
    axs[0].plot(U_right)
    axs[1].plot(U)
    axs[1].plot(U_left)
    plt.show()


def linspace(a,b,N,dtype):
    return tf.linspace(tf.constant(a,dtype),tf.constant(b,dtype),N)


def common_part(W, pad_size):
    if isinstance(pad_size, tuple) or isinstance(pad_size, list):
        pad_left = pad_size[0]
        pad_right = pad_size[1]
    else:
        pad_left = pad_size
        pad_right = pad_size


    assert pad_left < W.shape[-1] and pad_right < W.shape[-1], f"we must have pad_left < W.shape[-1] and pad_right < W.shape[-1]. Here:pad_left={pad_left},pad_right={pad_right} and W.shape[-1]={W.shape[-1]}"

    shape_left = W.shape[:-1] + (pad_left,)
    shape_right = W.shape[:-1] + (pad_right,)

    left_value = W[..., 0:1]
    right_value = W[..., -1:]

    return pad_left,pad_right,shape_left,shape_right,left_value,right_value


def dirichlet_padding(W, pad_size):
    pad_left,pad_right,shape_left,shape_right,left_value,right_value=common_part(W,pad_size)

    left_value_repeat = tf.ones(shape_left, dtype=W.dtype) * left_value
    right_value_repeat = tf.ones(shape_right, dtype=W.dtype) * right_value

    return tf.concat([left_value_repeat, W, right_value_repeat], axis=-1)


def linear_periodizing_padding(W: tf.Tensor, pad_size):

    pad_left,pad_right,shape_left,shape_right,left_value,right_value=common_part(W,pad_size)

    mean = tf.reduce_mean(W,axis=-1)[...,None]   #(a,b,1)

    amplitude_right=right_value-mean #(a,b,1)
    amplitude_left=left_value-mean #(a,b,1)

    right = linspace(0., 1, pad_right,W.dtype)[::-1]  # (p)
    right *= amplitude_right  # (a,b,p)
    right += mean

    left = linspace(0., 1, pad_left,W.dtype)  # (p)
    left *= amplitude_left
    left += mean

    res=tf.concat([left, W, right], axis=-1)

    return res



def neumann_padding(W: tf.Tensor, pad_size):

    pad_left,pad_right,shape_left,shape_right,left_value,right_value=common_part(W,pad_size)

    amplitude_right=right_value-W[...,-2:-1] #(a,b,1)
    amplitude_left=left_value-W[...,1:2] #(a,b,1)

    right = linspace(0., pad_right, pad_right,W.dtype)  # (p)
    right *= amplitude_right  # (a,b,p)
    right += right_value

    left = linspace(0., pad_left, pad_left,W.dtype)[::-1]  # (p)
    left *= amplitude_left
    left += left_value

    res=tf.concat([left, W, right], axis=-1)

    return res


def smooth_periodizing_padding(W: tf.Tensor, pad_size):

    #centrage du tenseur pour que le raccordement ait une plus petite amplitude
    #left_value = W[..., 0:1]
    #right_value = W[..., -1:]

    #Le centrage par la moyenne est moins sensible au changement de sampling d'un signal
    mean = tf.reduce_mean(W,axis=-1)[...,None]  #(left_value + right_value) / 2
    W -= mean

    pad_left,pad_right,shape_left,shape_right,left_value,right_value=common_part(W,pad_size)

    left = W[..., 1:pad_left + 1] - left_value
    right = W[..., -1 - pad_right:-1] - right_value
    left = -left[..., ::-1]
    right = -right[..., ::-1]
    left += left_value
    right += right_value

    left=tend_to_zero_left(left)
    right=tend_to_zero_right(right)

    res=tf.concat([left, W, right], axis=-1)
    res+=mean

    return res

def smooth_padding(W: tf.Tensor, pad_size):
    pad_left, pad_right, shape_left, shape_right, left_value, right_value = common_part(W, pad_size)

    left = W[..., 1:pad_left + 1] - left_value
    right = W[..., -1 - pad_right:-1] - right_value
    left = -left[..., ::-1]
    right = -right[..., ::-1]
    left += left_value
    right += right_value

    return tf.concat([left, W, right], axis=-1)


# uniquement le reflexive quand on veut qu'il préserve la positivité
def make_positive(A):
    return  tf.nn.elu(A - 1) + 1

def test_function_to_keep_positivity():
    x = tf.linspace(-3, 3, 100)
    y = tf.nn.elu(x - 1) + 1
    z = tf.nn.relu(x)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()

def zero_padding(W, pad_size):
    if isinstance(pad_size, tuple) or isinstance(pad_size, list):
        pad_left = pad_size[0]
        pad_right = pad_size[1]
    else:
        pad_left = pad_size
        pad_right = pad_size

    shape_left = W.shape[:-1] + (pad_left,)
    shape_right = W.shape[:-1] + (pad_right,)

    left_value_repeat = tf.zeros(shape_left,dtype=W.dtype)
    right_value_repeat = tf.zeros(shape_right,dtype=W.dtype)
    return tf.concat([left_value_repeat, W, right_value_repeat], axis=-1)


def periodic_padding(W, pad_size):

    if isinstance(pad_size, tuple) or isinstance(pad_size, list):
        pad_left = pad_size[0]
        pad_right = pad_size[1]
    else:
        pad_left = pad_size
        pad_right = pad_size

    assert pad_left < W.shape[-1] and pad_right < W.shape[-1]

    #ici c'est inversé
    left = W[..., :pad_right]
    right = W[..., -pad_left:]
    return tf.concat([right, W, left], axis=-1)


def test_periodizing():
    t=tf.linspace(0., 50, 100)
    U = tf.sin(t)+t/3-6
    U_mod = smooth_periodizing_padding(U,pad_size=99)
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(U)
    axs[1].plot(U_mod)

    plt.show()



def test_pad_1d():
    dtype=tf.float32
    x = linspace(0., 1, 110, dtype)
    y = tf.sin(x * 2 * np.pi) + (2 * x + 1)
    W = tf.ones([1,1,1],dtype)*y[None,:,None] #axis=1

    pad_size = [60,50]

    for kind in padding_kind.get_all_function():
        W_=pad_1d(W,kind,pad_size,axis=1)
        plt.plot(W_[0,:,0],label=kind)


    plt.legend(bbox_to_anchor=(0, 0.9),loc='center left')
    plt.show()




def pad_1d(W,kind,pad_size,axis,preserve_positivity=False):
    try:
        pad_func=padding_kind.get_one_function(kind)#kind_dict[kind]
    except KeyError:
        raise Exception(f"the kind:{kind} is not allowed. Allowed kinds are {padding_kind.key_to_func.keys()}")

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

    if preserve_positivity:
        W=make_positive(W)
    return W



def pad_nd(W,kind,pad_sizes,axes,preserve_positivity=False):
    assert  isinstance(kind,str), "kind must be a string in the following list:"+str(padding_kind.key_to_func.keys())
    assert isinstance(axes,tuple) or isinstance(axes,list), "axes must be a list or a tupe"
    assert isinstance(pad_sizes,tuple) or isinstance(pad_sizes,list), "pad_sizes must be a list or a tupe"

    assert len(pad_sizes)==len(axes)

    for pad_size,axis in zip(pad_sizes,axes):
        W=pad_1d(W,kind,pad_size,axis,preserve_positivity)

    return W

def unpad_nd(W,pad_sizes,axes):
    for pad_size,axis in zip(pad_sizes,axes):

        if isinstance(pad_size, tuple) or isinstance(pad_size, list):
            pad_left = pad_size[0]
            pad_right = pad_size[1]
        else:
            pad_left = pad_size
            pad_right = pad_size

        size_before_unpad=W.shape[axis]
        size_after=size_before_unpad-pad_left-pad_right

        W=slice_at_given_axis(W,pad_left,size_after,axis)

    return W



def slice_at_given_axis(W,begin,size,axis):
    beg=[0 for _ in W.shape]
    sizes=[i for i in W.shape]
    beg[axis]=begin%W.shape[axis]
    sizes[axis]=size
    res=tf.slice(W,beg,sizes)
    return res


def test_slice_at_given_axis():
    U=tf.random.uniform(shape=[5,6,7,8])
    U_slice=slice_at_given_axis(U,1,3,axis=1)
    print("before",U.shape)
    print("after",U_slice.shape)



def test_pad_unpad():
    U=tf.random.uniform(shape=[5,6,7,8,9])

    pad_sizess=[[1,2,3],[(1,2),(2,1),(3,0)],[(1,2),3,(4,4)]]
    axess = [[0, 2, 4],[1,2,4],[0,1,4]]

    for pad_sizes,axes in zip(pad_sizess,axess):
        U_pad=pad_nd(U,"zero_padding",pad_sizes,axes)
        U_unpad=unpad_nd(U_pad,pad_sizes,axes)
        print("shape are the same:",U.shape,U_unpad.shape)
        assert U.shape==U_unpad.shape
        #tensord are the same
        assert tf.reduce_mean(tf.abs(U-U_unpad))<1e-6


def test_pad_nd():
    xmin,xmax=0,2
    ymin,ymax=0,1
    dtype=tf.float32
    x=linspace(0.,2,40,dtype)
    y=linspace(0.,1,40,dtype)
    xx,yy=tf.meshgrid(x,y)
    z=tf.sin(10*xx)+yy

    keys=padding_kind.key_to_func.keys()
    nb=len(keys)
    def one(reverse):
        fig,axs=plt.subplots(nb,1,figsize=(5,3*nb))
        axes=[1,0] if reverse else [0,1]
        for i, key in enumerate(keys):
            W=pad_nd(z,key,[[10,20],[10,20]],axes=axes)
            axs[i].imshow(W,cmap="jet",origin="lower",extent=[xmin,xmax,ymin,ymax])
            axs[i].set_title(key)
        plt.tight_layout()

    one(False)
    one(True)

    plt.show()




padding_kind=Padding_kind()




if __name__=="__main__":
    #test_tend_to_value()
    #test_pad_1d()
    #test_periodizing()
    test_pad_nd()

    #test_slice_at_given_axis()


    test_pad_unpad()