pp=print
import numpy as np
import tensorflow as tf
import time



def create_permut(axes,dim):
    """
    Crée une permutation qui renvoie tous les indices de 'axes' à la fin.
    Si un indice est déjà à la fin, il n'est pas bougé
    @param axes:
    @param dim:
    @return:
    """
    last_index=np.arange(dim-len(axes),dim)
    to_move=[]
    place=[]
    for i in axes:
        if i<dim-len(axes):
            to_move.append(i)
    for j in last_index:
        if j not in axes:
            place.append(j)
    res=np.arange(dim)
    for i,j in zip(to_move,place):
        res[i]=j
        res[j]=i
    return res

def create_several_permut(axess,dim):
    res=[]
    compo_inv=np.arange(dim)
    for axes in axess:
        perm_loc=create_permut(axes,dim)
        perm=do_permut(compo_inv,perm_loc)
        res.append(perm)
        compo_inv=do_permut(compo_inv,perm)
    return res,compo_inv

def test_create_several_permut():
    axess=[[1,3],[0,3],[3,1,2],[2]]
    print("axess",axess)
    res,compo_inv=create_several_permut(axess,4)

    liste=[0,1,2,3]
    print("initial", liste)
    for perm,axes in zip(res,axess):
        liste = do_permut(liste, perm)
        nb=len(axes)
        print(liste)
        assert set(liste[-nb:])==set(axes)
    back=do_permut(liste,compo_inv)
    print("retour",back)


def do_permut(liste,permut1):
    res=[]
    assert len(permut1)==len(liste)
    for j in permut1:
        res.append(liste[j])
    return res

def test_create_permut():
    permut=create_permut([0,2],4)
    print("permut",permut)

    liste=np.arange(4)
    liste_=do_permut(liste,permut)
    liste__=do_permut(liste_,permut)
    print(liste)
    print(liste_)
    print(liste__)

def test_double_permut():
    #remarque: si les deux ensembles d'indices sont disjoints, alors les permutations commutent
    permut1=create_permut([0,2],4)
    permut2=create_permut([1,2],4)

    liste=["0","1","2","3"]
    print("initial",liste)
    for perm in [permut1,permut2]:
        liste=do_permut(liste,perm)
        print(liste)

    for perm in [permut2,permut1]:
        liste = do_permut(liste, perm)
        print(liste)

def fft_nd(W,axes):
    return Fft_nd(axes)(W)
def ifft_nd(W,axes):
    return Fft_nd(axes,inverse=True)(W)



class Fft_nd:
    def __init__(self,axes,axes_grouping=None,inverse=False):
        self.axes=axes
        self.inverse=inverse

        if axes_grouping is None:
            axes_grouping=[]
            nb=len(axes)
            q=nb//3
            r=nb%3
            for _ in range(q):
                axes_grouping.append(3)
            if r>0:
                axes_grouping.append(r)

        assert sum(axes_grouping) == len(axes)
        for nb in axes_grouping:
            assert nb<=3,"max 3"

        self.axess=[]
        current=0
        for n in axes_grouping:
            self.axess.append(axes[current:current+n])
            current+=n

        self.permuts=None
        self.comme_back=None

    def __call__(self,W):
        if self.permuts is None:
            self.permuts, self.comme_back = create_several_permut(self.axess, len(W.shape))

        for i in range(len(self.axess)):
            axes=self.axess[i]
            size=len(axes)
            permut=self.permuts[i]
            W=tf.transpose(W,permut)


            if self.inverse:
                if size==1:
                    W=tf.signal.ifft(W)
                elif size==2:
                    W=tf.signal.ifft2d(W)
                else:
                    W=tf.signal.ifft3d(W)
            else:
                if size == 1:
                    W = tf.signal.fft(W)
                elif size == 2:
                    W = tf.signal.fft2d(W)
                else:
                    W = tf.signal.fft3d(W)

        W=tf.transpose(W,self.comme_back)
        return W




"""
Test automatiques
"""

def test_fft_nd():
    #fft_nd=Fft_nd([1,2,3,4,5,6],[1,2,3])

    W=tf.random.uniform([2,3,4])
    W=tf.complex(W,W)
    F1=tf.signal.fft3d(W)

    F2=Fft_nd([0,1,2])(W)
    F3=Fft_nd([0,1,2],[2,1])(W)
    F4=Fft_nd([0,1,2],[1,1,1])(W)
    F5=Fft_nd([0,1,2],[1,2])(W)

    def mse_fn(A):
        return tf.reduce_mean(tf.square(A))
    for F in [F2,F3,F4,F5]:
        mse=mse_fn(F-F1)
        print(mse.numpy())
        assert tf.abs(mse)<1e-12



if __name__=="__main__":
    #test_create_permut()
    #test_double_permut()
    #test_create_several_permut()
    test_fft_nd()


