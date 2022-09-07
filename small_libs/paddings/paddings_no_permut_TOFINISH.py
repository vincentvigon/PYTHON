
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
pp=print



def tend_to_zero_right(U):
    N=U.shape[-1]
    t=tf.linspace(0.,2,N)
    t=tf.cast(t,U.dtype)
    decrease=tf.exp(-t**4)
    return U*decrease
def tend_to_zero_left(U):
    N=U.shape[-1]
    t=tf.linspace(-2,0,N)
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


def collect_at_given_axis(W,index,axis):
    beg=[0 for _ in W.shape]
    size=[i for i in W.shape]
    beg[axis]=index%W.shape[axis]
    size[axis]=1
    res=tf.slice(W,beg,size)
    return res

def slice_at_given_axis(W,begin,size,axis):
    beg=[0 for _ in W.shape]
    sizes=[i for i in W.shape]
    beg[axis]=begin%W.shape[axis]
    sizes[axis]=size
    pp(beg)
    pp(sizes)
    res=tf.slice(W,beg,sizes)
    return res

def reverse_at_given_axis(W,axis):
    size=W.shape[axis]
    indices=tf.range(start=0,limit=size,delta=-1)
    return tf.gather(W,indices=indices)


def test_collect_at_given_axis():
    W = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    #W.shape=(3, 2, 3)
    def mae(a,b):
        return tf.reduce_sum(tf.abs(a-b))

    t1=collect_at_given_axis(W,0,0)
    assert mae(t1,W[0:1,:,:])==0
    t2=collect_at_given_axis(W,0,1)
    assert mae(t2,W[:,0:1,:])==0
    t3=collect_at_given_axis(W,0,-1)
    assert mae(t3,W[:,:,0:1])==0

    t1=collect_at_given_axis(W,1,0)
    assert mae(t1,W[1:2,:,:])==0
    t2=collect_at_given_axis(W,1,1)
    assert mae(t2,W[:,1:2,:])==0
    t3=collect_at_given_axis(W,1,-1)
    assert mae(t3,W[:,:,1:2])==0

    t1=collect_at_given_axis(W,-1,0)
    assert mae(t1,W[-1:,:,:])==0
    t2=collect_at_given_axis(W,-1,1)
    assert mae(t2,W[:,-1:,:])==0
    t3=collect_at_given_axis(W,-1,-1)
    assert mae(t3,W[:,:,-1:])==0


def compute_shapes(W,pad_sizes,axis):

    if isinstance(pad_sizes, tuple) or isinstance(pad_sizes, list):
        pad_left = pad_sizes[0]
        pad_right = pad_sizes[1]
    else:
        pad_left = pad_sizes
        pad_right = pad_sizes

    assert pad_left < W.shape[-1] and pad_right < W.shape[-1]

    shape_left = list(W.shape)
    shape_left[axis]=1
    shape_right = list(W.shape)
    shape_right[axis]=1

    right_value = collect_at_given_axis(W,-1,axis)# si axis=-1, cela revient à faire W[..., -1]
    left_value = collect_at_given_axis(W,0,axis)  # si axis=-1, cela revient à faire W[..., -0]

    left_value_repeat = tf.ones(shape_left ,dtype=W.dtype) * left_value
    right_value_repeat = tf.ones(shape_right ,dtype=W.dtype) * right_value

    return shape_left,shape_right,pad_left,pad_right,left_value,right_value,left_value_repeat,right_value_repeat


def neumann_padding(W: tf.Tensor, pad_sizes,axis):
    shape_left,shape_right,pad_left,pad_right,left_value,right_value,left_value_repeat,right_value_repeat=compute_shapes(W, pad_sizes,axis)
    return tf.concat([left_value_repeat, W, right_value_repeat], axis=axis)


def smooth_periodizing_padding(W: tf.Tensor, pad_sizes,axis):
    shape_left, shape_right, pad_left, pad_right, left_value, right_value,left_value_repeat,right_value_repeat = compute_shapes(W, pad_sizes, axis)

    mid_value = (right_value+left_value)/2
    W-=mid_value

    #left = W[..., 1:pad_left + 1] - left_value_repeat
    left = slice_at_given_axis(W,1,pad_left,axis)
    #right = W[..., -1 - pad_right:-1] - right_value_repeat
    pp("W",W.shape,"pad_right",pad_right,"axis",axis)
    right= slice_at_given_axis(W,-1,pad_right,axis)
    left = -reverse_at_given_axis(left,axis)
    right = -reverse_at_given_axis(right,axis)
    left += left_value_repeat
    right += right_value_repeat

    left=tend_to_zero_left(left)
    right=tend_to_zero_right(right)

    res=tf.concat([left, W, right], axis=-1)
    res+=mid_value

    return res


def test_padding_2d_small():
    W=tf.constant([[1.1,1.2,1.3],[2.1,2.2,2.3]])
    print("tenseur initial\n",W)

    for pad_function,name in zip([neumann_padding, smooth_periodizing_padding],["dirichlet_padding", "smooth_periodizing_padding"]):
        print(name)
        W0=pad_function(W,[1,2],0)
        print("padding axe 0",W0)
        W1 = pad_function(W, [1, 2], 1)
        print("padding axe 1", W1)
        print()



if __name__=="__main__":
    #test_collect_at_given_axis()
    test_padding_2d_small()
