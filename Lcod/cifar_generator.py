#python scripts
__author__='Du Jiawei'
#Descrption:
import numpy as np
from sklearn.utils import shuffle as shuf
from keras.datasets import cifar10
class CifarProblemGenerator():
    def __init__(self, D, lmbd, batch_size=100, dir_mnist='save_exp_cifar',seed=0):
        self.D = np.array(D)
        self.K, self.p = D.shape
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.patch_size = int(np.sqrt(self.p))
        self.dict = unpickle("/data/home/dujw/darse/save_exp/cifar/data_batch_1")
        self.seed = seed
        # self.im =  self.dict['data'].astype(np.float32)/255
        # self.label = self.dict['labels']
        (im,self.label),(x_t,y_t) = cifar10.load_data()
        im = im.astype(np.float32)/255
        self.im = im.reshape(im.shape[0],-1)
    def get_batch(self, N=None,shuffle=True):
        if N is None:
            N = self.batch_size
        # im = self.im
        # batch = shuf(im,random_state=np.random.randint(1,10000))
        # X = batch[:N].reshape(N, -1)
        X,label = self.get_batch_with_label(N,shuffle)
        z = np.zeros((N, self.K))
        return X, z, z, label,self.lmbd
    def get_truth(self,N):
        return self.dict['data'][:N]
    def get_batch_with_label(self,N=None,shuffle=True):
        from keras.utils import np_utils
        if N is None:
            N = self.batch_size
        im = self.im
        if shuffle ==True:
            batch,label = shuf(im,self.label,random_state=np.random.randint(1,10000))
        else:
            batch,label = im,self.label
        X = batch[:N].reshape(N, -1)
        label =label[:N]
        label = np_utils.to_categorical(label,10)
        return X,label
    def get_test(self,N=None):
        X, z, z, label,self.lmbd  = self.get_batch(N=N)
        return X, z, z,self.lmbd  



def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def create_dictionary_dl(lmbd, K=100, N=10000, dir_mnist='save_exp_cifar'):
    import os.path as osp
    dict = unpickle("/data/home/dujw/darse/save_exp/cifar/data_batch_1")
    fname = osp.join(dir_mnist, "D_mnist_K{}_lmbd{}.npy".format(K, lmbd))
    if osp.exists(fname):
        D = np.load(fname)
    else:
        from sklearn.decomposition import DictionaryLearning
        X = dict['data'].astype(np.float32)/255
        dl = DictionaryLearning(K, alpha=lmbd*N, fit_algorithm='cd',n_jobs=-1, verbose=1)
        dl.fit(X)
        D = dl.components_.reshape(K, -1)
        np.save(fname, D)
    return D 
if __name__ == '__main__':
    D = create_dictionary_dl(0.1,K=256)

