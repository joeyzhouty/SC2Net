import numpy as np
from glob import glob
from scipy.misc import imresize
from tensorflow.examples.tutorials.mnist import input_data
import cv2
from deepLearningModel.imagenet_utils import preprocess_input
from deepLearningModel.resnet50 import ResNet50
from cifar_generator import  unpickle
class MnistProblemGenerator(object):
    """A simple problem to test the capability of LISTA

    Parameters
    ----------
    D: array-like, [K, p]
        dictionary for the generation of problems. The shape should be
        [K, p] where K is the number of atoms and p the dimension of the
        output space
    lmbd: float
        sparsity factor for the computation of the cost
    """
    def __init__(self, D, lmbd, batch_size=100, dir_mnist='save_exp/mnist',
                 seed=None):
        super(MnistProblemGenerator,self).__init__()
        self.D = np.array(D)
        self.K, self.p = D.shape
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.patch_size = int(np.sqrt(self.p))

        # self.model = ResNet50(weights='imagenet',include_top=False) 
        # Load training images
        self.mnist = input_data.read_data_sets(dir_mnist, one_hot=True)
        self.dict = unpickle("save_exp/mnist/feature.pkl")
        self.feat = self.dict['feature']
        self.label = self.dict['label']




    def get_feature_batch(self,N):
        from sklearn.utils import shuffle as shuf
        batch,label = shuf(self.feat,self.label,random_state=np.random.randint(1,10000))
        batch = batch[:N].reshape(N,-1)
        label = label[:N]
        return batch,label
    def get_batch(self, N=None,shuffle=True):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size

        # Take mnist 17x17
        # im = self.mnist.train.next_batch(N)[0].reshape(N, 28, 28)
        # im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
              # for a in im]
        z = np.zeros((N, self.K))
        # X = np.array(im).reshape(N, -1)
        X,label = self.get_batch_with_label(N,shuffle)
        # X = feat_extract(self.model,im)
        # X = self.get_feature_batch(N)[0]
        return X, z, z, label,self.lmbd

    def get_batch_with_label(self, N=None,shuffle=True):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size
        # Take mnist 17x17
        im,label = self.mnist.train.next_batch(N)
        im = im.reshape(N,28,28)
        # im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
              # for a in im]
        z = np.zeros((N, self.K))
        X = np.array(im).reshape(N, -1)
        # X = feat_extract(self.model,im)
        # X,label = self.get_feature_batch(N)
        return X, label 


    def get_test(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size

        # Take mnist 17x17
        im = self.mnist.test.next_batch(N)[0].reshape(N, 28, 28)
        # im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
              # for a in im]
        z = np.zeros((N, self.K))
        # X = np.array(im).reshape(N, -1)
        X = self.get_feature_batch(N)[0]
        return X, z, z, self.lmbd

    def lasso_cost(self, z, sig):
        '''Cost of the point z for a problem with sig'''
        residual = sig - z.dot(self.D)
        Er = np.sum(residual*residual, axis=1)/2
        return Er.mean() + self.lmbd*abs(z).sum(axis=1).mean()


def feat_extract(model,img):
    newimg = []
    for each in img:
        each = np.repeat(cv2.resize(each,(224,224)).reshape(224,224,1),3,axis=2)
        newimg.append(each)

    newimg = np.array(newimg)
    feat = model.predict(newimg)
    feat = np.reshape(feat,(feat.shape[0],-1))
    return feat

def create_dictionary_dl(lmbd, K=100, N=10000, dir_mnist='save_exp/mnist'):

    import os.path as osp
    fname = osp.join(dir_mnist, "D_mnist_K{}_lmbd{}.npy".format(K, lmbd))
    if osp.exists(fname):
        D = np.load(fname)
    else:
        from sklearn.decomposition import DictionaryLearning
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        im = mnist.train.next_batch(N)[0]
        im = im.reshape(N, 28, 28)
        # im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
              # for a in im]
        X = np.array(im).reshape(N, -1)
        # model = ResNet50(weights='imagenet',include_top=False)
        # X = feat_extract(model,im)
        print(X.shape)

        dl = DictionaryLearning(K, alpha=lmbd*N, fit_algorithm='cd',
                                n_jobs=-1, verbose=1)
        dl.fit(X)
        D = dl.components_.reshape(K, -1)
        np.save(fname, D)
    return D
if __name__ == '__main__':
    D = create_dictionary_dl(0.1)
    pb = MnistProblemGenerator(D,0.1)
    test_x,test_y = pb.get_batch_with_label(8000)
    test_y = np.argmax(test_y,axis=1)
    from sklearn.linear_model import LogisticRegression as lgr
    from sklearn.metrics import accuracy_score
    lgrs = lgr()
    lgrs.fit(test_x[:6000],test_y[:6000])
    y_pre = lgrs.predict(test_x[6000:])
    print accuracy_score(test_y[6000:],y_pre)



