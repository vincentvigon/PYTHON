
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from fourier_neural_operator.FNO_2d import FNO2d
pp=print

def visu_img(imgs):
  n_imgs = len(imgs)
  fig, axs = plt.subplots(1, n_imgs)
  if n_imgs == 1:
    axs = [axs]
  for i in range(n_imgs):
    axs[i].imshow(imgs[i], interpolation="nearest", cmap="gray")
  plt.show()


def visu_flow(flows):
  n_flows = len(flows)
  fig, axs = plt.subplots(1, n_flows)
  if n_flows == 1:
    axs = [axs]
  for i in range(n_flows):
    u, v = flows[i][0,:,:,0],flows[i][0,:,:,1]
    axs[i].quiver(v,u,)#ici j'ai inversé
  plt.show()


def grid_sample_2d(inp, grid):

    in_shape = tf.shape(inp)

    in_h = in_shape[1]
    in_w = in_shape[2]

    # Find interpolation sides
    i, j = grid[..., 0], grid[..., 1]
    i = tf.cast(in_h - 1, grid.dtype) * (i + 1) / 2
    j = tf.cast(in_w - 1, grid.dtype) * (j + 1) / 2
    i_1 = tf.maximum(tf.cast(tf.floor(i), tf.int32), 0)
    i_2 = tf.minimum(i_1 + 1, in_h - 1)
    j_1 = tf.maximum(tf.cast(tf.floor(j), tf.int32), 0)
    j_2 = tf.minimum(j_1 + 1, in_w - 1)

    # Gather pixel values
    n_idx = tf.tile(tf.range(in_shape[0])[:, tf.newaxis, tf.newaxis], tf.concat([[1], tf.shape(i)[1:]], axis=0))
    q_11 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_1], axis=-1))
    q_12 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_2], axis=-1))
    q_21 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_1], axis=-1))
    q_22 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_2], axis=-1))

    # Interpolation coefficients
    di = tf.cast(i, inp.dtype) - tf.cast(i_1, inp.dtype)
    di = tf.expand_dims(di, -1)
    dj = tf.cast(j, inp.dtype) - tf.cast(j_1, inp.dtype)
    dj = tf.expand_dims(dj, -1)

    # Compute interpolations
    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj

    return q_ij

def test_grid_sample():
    im_grid_i, im_grid_j = np.meshgrid(np.arange(20), np.arange(30), indexing='ij')
    im = im_grid_i + im_grid_j
    im = im / im.max()
    im = np.stack([im,tf.sin(5*im),tf.sin(3*im)], axis=-1)
    # Test grid 1: complete image
    grid1 = np.stack(np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 0, 18), indexing='ij'), axis=-1)
    # Test grid 2: lower right corner
    grid2 = np.stack(np.meshgrid(np.linspace(-1, 1, 13), np.linspace(0, 1, 17), indexing='ij'), axis=-1)

    print(im.shape)
    print(grid1.shape)

    # Run
    res1 = grid_sample_2d(im[None,:,:,:], grid1[None,:,:,:])[0,:,:,:]
    res2 = grid_sample_2d(im[None,:,:,:], grid2[None,:,:,:])[0,:,:,:]

    print(res1.shape)
    print(res2.shape)

    # Plot image and sampled grids
    plt.figure()
    plt.title("initial image")
    plt.imshow(im)
    plt.figure()
    plt.title("left part")
    plt.imshow(res1)
    plt.figure()
    plt.title("right part")
    plt.imshow(res2)

    plt.show()

def import_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    np.save("x_train",x_train)
    np.save("y_train",y_train)
    np.save("x_test",x_test)
    np.save("y_test",y_test)

def get_data_preprocess():
    def get_data():
        x_train = np.load("x_train.npy")
        y_train = np.load("y_train.npy")
        x_test = np.load("x_test.npy")
        y_test = np.load("y_test.npy")
        return (x_train.astype(np.float32), y_train), (x_test.astype(np.float32), y_test)

    (x_train_all, y_train_all), (x_test_all, y_test_all)=get_data()
    # extract all 3s
    digit = 3

    x_train = x_train_all[y_train_all == digit, ...]
    x_test = x_test_all[y_test_all == digit, ...]

    nb_val = 1000  # keep 1,000 subjects for validation
    x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
    x_train = x_train[:-nb_val, ...]

    # %% normalization

    x_train = x_train/ 255
    x_val = x_val / 255
    x_test = x_test / 255

    pad_amount = ((0, 0), (2, 2), (2, 2))

    # fix data
    x_train = np.pad(x_train, pad_amount, 'constant')
    x_val = np.pad(x_val, pad_amount, 'constant')
    x_test = np.pad(x_test, pad_amount, 'constant')

    # verify
    x_train, x_val, x_test=x_train[:,:,:,None],x_val[:,:,:,None],x_test[:,:,:,None]
    print('shape of training data', x_train.shape)

    return x_train,x_val,x_test


class DataDealer:
    def __init__(self):
        self.x_train, _,_= get_data_preprocess()
        self.size=len(self.x_train)
        self.image_shape=(self.x_train.shape[1],self.x_train.shape[2])
        self.batch_size=64


    def get_iterator_for_one_epoch(self):
        indices=tf.random.shuffle(range(self.size))
        data_source=self.x_train[indices]
        indices=tf.random.shuffle(range(self.size))
        data_target=self.x_train[indices]

        nb_batch=self.size//self.batch_size

        for i in range(nb_batch):
            sl=slice(i*self.batch_size,(i+1)*self.batch_size)
            yield data_source[sl,:,:],data_target[sl,:,:]


class SpatialTransformer:
    """
    N-D Spatial Transformer
    """
    def __init__(self, image_shape):
        super().__init__()
        h, w=image_shape
        self.img_h, self.img_w = image_shape
        ones=np.ones([h,w])
        hh = tf.range(h, dtype=tf.float32)[:,None]*ones
        ww = tf.range(w, dtype=tf.float32)[None, :]*ones
        grid = tf.stack([hh,ww], axis=2)
        self.grid = grid[None,:,:,:]

        # create sampling grid
        # vectors = [tf.range(0, s,dtype=tf.float32) for s in image_shape]
        # grids = tf.meshgrid(vectors)
        # grid = tf.stack(grids)
        # self.grid = grid[None,...]


    def forward(self, src, flow):
        assert src.shape[:-1]==flow.shape[:-1],f"src.shape[:-1]={src.shape[:-1]} != flow.shape={flow.shape[:-1]}"
        assert flow.shape[3]==2,f"flow.shape={flow.shape}, but the 4th element must be 2"

        # new locations
        new_locs = self.grid + flow
        # need to normalize grid values to [-1, 1] for resampler
        new_locs_0=(new_locs[:,:,:,0]/(self.img_h-1)-0.5)*2
        new_locs_1=(new_locs[:,:,:,1]/(self.img_w-1)-0.5)*2


        new_locs=tf.stack([new_locs_0,new_locs_1],axis=3)
        new_locs=tf.clip_by_value(new_locs,-1,1)



        return grid_sample_2d(src, new_locs)

def test_SpatialTransformer():
    h,w=32,32

    hh=tf.range(h,dtype=tf.float32)[:,None]
    ww=tf.range(w,dtype=tf.float32)[None,:]
    flow_h=(hh+ww)*0.2
    flow_w=(hh+ww)*0.1
    flow=tf.stack([flow_h,flow_w],axis=-1)
    flow=flow[None,:,:,:]

    visu_flow([flow])

    x, _,_=get_data_preprocess()
    transformer=SpatialTransformer([h,w])
    x=x[:1,:,:,:]
    print("x,flow",x.shape,flow.shape)
    x_trans=transformer.forward(x,flow)

    visu_img([x[0,:,:,0],x_trans[0,:,:,0]])


class Morph_agent:

    def __init__(self):
        super().__init__()

        self.data_dealer=DataDealer()

        self.model = FNO2d(modes=10,width=20,out_channels=2)
        self.transformer = SpatialTransformer(image_shape=self.data_dealer.image_shape)
        self.optimizer=tf.keras.optimizers.Adam()

        self.losses_target=[]
        self.losses_phi=[]
        self.losses=[]



    def train_epochs(self,nb_epochs):
        for i in range(nb_epochs):
            ite=self.data_dealer.get_iterator_for_one_epoch()
            for source,target in ite:
                loss,loss_target,loss_phi=self.training_step(source,target)
                print(loss.numpy())

                self.losses.append(loss.numpy())
                self.losses_target.append(loss_target.numpy())
                self.losses_phi.append(loss_phi.numpy())


    def forward(self, source, target):


        X=tf.concat([source, target], axis=3)
        flow = self.model.call(X)
        y_source = self.transformer.forward(source, flow)
        return y_source, flow


    @tf.function
    def training_step(self,source, target):

        with tf.GradientTape() as tape:
            y_source, flow = self.forward(source, target)
            loss_target = tf.reduce_mean((y_source - target)**2)
            loss_phi = 0#tf.reduce_mean(tf.square(flow)) * 2
            loss = loss_target + loss_phi

        tv=self.model.trainable_variables
        gradient=tape.gradient(loss,tv)
        self.optimizer.apply_gradients(zip(gradient,tv))

        return loss,loss_target,loss_phi



def main():
    agent=Morph_agent()
    agent.train_epochs(1)
    fig,ax=plt.subplots(1,1)
    ax.plot(agent.losses,legend="total")
    ax.plot(agent.losses_phi,legend="phi")
    ax.plot(agent.losses_target,legend="total")

    ax.legend()

    plt.show()



if __name__=="__main__":
    #test_grid_sample()
    #test_SpatialTransformer()
    main()





