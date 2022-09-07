import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import time
pp=print



def create_bank_of_points(n,use_Halton=True):
    if use_Halton:
        # ici scramble=True n'améliore pas beaucoup les choses sur 300 points, ni sur 3000
        sampler = qmc.Halton(d=2,scramble=False)  # en recréant un nouveau sampler pour chaque cellule, on n'améliore pas la discrépence
        bank =  sampler.random(n=n)
    else:
        bank =  np.random.uniform(0,1,[n, 2]).astype(np.float32)

    np.save("./bank.npy",bank)



class Preference_sampling:

    def __init__(self,a_range,b_range,nb,min_sample_per_cell,nb_samples,value_smoothing=True):
        """
        :param a_range: min, max de la première coordonnée
        :param b_range: min, max de la seconde coordonnée
        :param nb:  nombre de cellule par coordonnée
        :param min_sample_per_cell: nombre de point minimum par cellule, indépendamment de la valeur de la fonction
        """
        self.a_range=a_range
        self.b_range=b_range
        self.nb=nb
        self.min_sample_per_cell=min_sample_per_cell
        self.nb_samples=nb_samples

        self.centers=self.make_centers()
        self.nb_centers=len(self.centers)

        self.value_smoothing=value_smoothing

        """
        Ici on tire tous les points d'un coup. On pourrait aussi les tirer au fur et à mesure, c'est un 
        peu moins performant pour les grosses tailles de nb_samples.
        """
        self._bank = tf.constant(np.load("./bank.npy"),dtype=np.float32)

        nb_point_required=1000+nb_samples+self.nb[0]*self.nb[1]*self.min_sample_per_cell
        print(f"nb point used:{nb_point_required}")
        assert nb_point_required< self._bank.shape[0], f"please, increase the size of your bank point to {nb_point_required}"

        scale=tf.stack([self.db,self.da])
        shift=tf.stack([-0.5*self.db, -0.5 * self.da])
        self._bank *= scale
        self._bank += shift

        self.mask = tf.ones([3, 3, 1, 1])


    def make_centers(self):

        self.a_border=tf.linspace(self.a_range[0]*1.,self.a_range[1], self.nb[0] + 1)
        a=self.a_border[:-1]
        self.da= a[1] - a[0]
        a= a + self.da / 2

        self.b_border=tf.linspace(self.b_range[0]*1., self.b_range[1], self.nb[1] + 1)
        b = self.b_border[:-1]
        self.db = b[1] - b[0]
        b = b + self.db / 2

        aa,bb=tf.meshgrid(a,b)

        aa_flat=tf.reshape(aa,[-1])
        bb_flat=tf.reshape(bb,[-1])
        return tf.stack([bb_flat,aa_flat],axis=1)



    def sample(self,func):
        values_at_center = func(self.centers)  # (N)
        #current=tf.cast(tf.random.uniform(maxval=1000,shape=[]),dtype=tf.int32)
        current=tf.constant(np.random.randint(0,1000),dtype=tf.int32)
        return self.sample_acc(values_at_center,current)


    @tf.function
    def sample_acc(self,values_at_center,current):
        print(f"traçage de la fonction sample, current={current}")

        if self.value_smoothing:
            values_matrix=tf.reshape(values_at_center,[1,self.nb[0],self.nb[1],1])
            C = tf.nn.conv2d(values_matrix, self.mask, [1, 1, 1, 1], padding="SAME")
            values_at_center=tf.reshape(C[0,:,:,0],[-1])

        assert len(values_at_center.shape)==1, f"values_at_center must be a vector, but its shape is {values_at_center.shape}"

        values_at_center-=tf.reduce_min(values_at_center)
        values_at_center/=tf.reduce_sum(values_at_center)

        samples=[]

        for i in range(len(values_at_center)):
            center=self.centers[i]
            value=values_at_center[i]
            n=tf.floor(value*self.nb_samples)+self.min_sample_per_cell
            n=tf.cast(n,tf.int32)

            #l=self._bank.shape[0]
            #if current%l>=(current + n)%l:
            #    current = current + n
            sample = self._bank[current:(current + n)]
            current=current+n
            sample+=center
            samples.append(sample)

        return  tf.concat(samples,axis=0)



def test():

    sampling=Preference_sampling([-1,1],[-1,1],[10,10],min_sample_per_cell=10,nb_samples=4000,value_smoothing=True)

    fonc=lambda X:tf.sin(5*X[:,0])
    fonc=lambda X:tf.cast(X[:,0]<0,tf.float32)

    centers=sampling.make_centers()


    values=fonc(centers)
    plt.xticks(sampling.a_border)
    plt.yticks(sampling.b_border)

    res = sampling.sample(fonc)

    plt.scatter(res[:,1],res[:,0],alpha=0.3)

    plt.scatter(centers[:, 1], centers[:, 0], c=values[:], cmap="jet")
    plt.colorbar()

    plt.show()



def test_perf():
    nb_samples = 50_000

    sampling=Preference_sampling([-1,1],[-1,1],[10,10],10,nb_samples)
    fonc=lambda X:X[:,1]


    sampling.sample(fonc)

    ti0=time.time()
    for _ in range(100):
        sampling.sample(fonc)

    duration=time.time()-ti0
    print(f"duration:",duration)

    #il faut 5 secondes sans l'accélérateur @tf.function
    assert duration<0.2
    #sur mon ordi le test_periodizing de perf oscille entre 0.13 et 0.15 (on a perdu 0.02 en faisant varier l'initialisation du "current")



def test_Haminton():
    sampler = qmc.Halton(d=2, scramble=False)
    batch=20
    for i in range(5):
        #sampler.fast_forward(i*batch)
        sample = sampler.random(n=batch)
        plt.scatter(sample[:,0],sample[:,1])

    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=batch*5)

    plt.scatter(sample[:, 0], sample[:, 1],marker="+")
    plt.show()



if __name__=="__main__":
    #create_bank_of_points(100_000)
    #test_Haminton()
    #test_periodizing()
    test_perf()