import numpy as np
from glob import glob
from scipy.misc import imresize
from tensorflow.examples.tutorials.mnist import input_data
import cv2
from cifar_generator import  unpickle
import pickle as pk
import os.path as osp
import os 
import random
class SyntheticProblemGenerator(object):

    def __init__(self, D=None,lmbd=0.1,m=100,n=20,d=2,batch_size=100, case = 0,save_exp='/home/dujw/darse/save_exp/synthetic',
                 seed=None):
        self.K, self.p =  m,n
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        if not osp.exists(osp.join(save_exp,"d{}_m{}_n{}_case{}.pkl".format(d,m,n,case))):
            syn = synthetic_generate(d=d,m=m,n=n,case=case)
            syn.save()


        self.dict = pk.load(open(osp.join(save_exp,"d{}_m{}_n{}.pkl".format(d,m,n)),"rb"))
        self.sparse = self.dict['x']
        self.rawdata = self.dict['y']
        self.sparse = self.sparse.reshape(len(self.sparse),-1)
        self.rawdata = self.rawdata.reshape(len(self.sparse),-1)
        self.phi = self.dict['phi']




    def get_feature_batch(self,N):
        from sklearn.utils import shuffle as shuf
        batch,label = shuf(self.feat,self.label,random_state=np.random.randint(1,10000))
        batch = batch[:N].reshape(N,-1)
        label = label[:N]
        return batch,label
    def get_batch(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size
        z = np.zeros((N, self.K))
        sample = random.sample(range(len(self.sparse)),N)
        X = self.rawdata[sample]
        X = X /np.std(X)
        label =np.random.random([N,10])
        return X, z, z, label,self.lmbd


    def get_batch_with_label(self, N=None,shuffle=True):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size
        if shuffle == True:
            sample = random.sample(range(len(self.sparse)),N)
            label = self.sparse[sample]
            X = self.rawdata[sample]
        else:
            label = self.sparse[:N]
            X = self.rawdata[:N]
        return X, label 


    def get_test(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size


        X, z, z, label,self.lmbd =self.get_batch(N)
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

def create_dictionary_dl(lmbd, d=2,m=100,n=20, N=10000,case=0,dir_mnist='/home/dujw/darse/save_exp/synthetic'):

    import os.path as osp
    fname = osp.join(dir_mnist, "D_synthetic_d{}_m{},n{},case{},lmbd{}.npy".format(d,m,n,case,lmbd))
    if osp.exists(fname):
        D = np.load(fname)
    else:
        from sklearn.decomposition import DictionaryLearning
        aa = SyntheticProblemGenerator(d=d,m=m,n=n)
        X = aa.get_batch(N)[0]
        K = m
        dl = DictionaryLearning(K, alpha=lmbd*N, fit_algorithm='cd',
                                n_jobs=-1, verbose=1)
        dl.fit(X)
        D = dl.components_.reshape(K, -1)
        np.save(fname, D)
    return D
class synthetic_generate(object):
    def __init__(self,d,N=100000,m=100,n=20,case=0,save_exp="/home/dujw/darse/save_exp/synthetic"):
        self.N = N
        self.case = case
        self.m = m
        self.n = n
        self.d = d
        self.save_exp = save_exp
        self.phi = self.getPhi()
        self.x, self.y = self.getXY()
    def getXY(self):
        N = self.N
        m = self.m
        n = self.n
        X = []
        Y = []
        for i in range(1,N+1):
            x = np.zeros([m,1])
            order = np.arange(m)
            np.random.shuffle(order)
            for j in range(self.d):
                x[order[j]] = self.getuniform()
            X.append(x)
            Y.append(np.matmul(self.phi,x))
        return np.array(X),np.array(Y)



    def save(self):
        import pickle as pk
        import os.path as osp 
        mydict = {'phi':self.phi,'x':self.x,'y':self.y}
        output = open(osp.join(self.save_exp,"d{}_m{}_n{}_case{}.pkl".format(self.d,self.m,self.n,self.case)),"wb")
        pk.dump(mydict,output)
        output.close()
    def getuniform(self,a=-1,b=1):
        result = np.random.uniform(a,b)
        while result < 0.3*b and result > 0.3*a:
            result = np.random.uniform(a,b)
        return result
    def getPhi(self):
        n = self.n
        m = self.m
        phi = np.zeros([n,m])
        if self.case == 0:
            for i in range(1,n+1):
                u = np.random.normal(0,scale=1,size=(n,1))
                v = np.random.normal(0,scale=1,size=(m,1))
                res = np.matmul(u,np.transpose(v))
                phi = phi + res/(i*i)

        if self.case == 1:
            A = np.random.normal(0,scale=1,size=(n,m))
            R = np.matmul(np.random.normal(0,scale=1,size=(n,5)),np.random.normal(0,scale=1,size=(5,m)))
            phi = 0.3*A+1*R
        if self.case == 2:
            Dlist = [] 
            for i in range(50):
                u = np.random.normal(0,scale=1,size=(n,1))
                v = np.random.normal(0,scale=1,size=(1,m/50))
                Ai = np.random.normal(0,scale=1,size=(n,m/50))
                Dlist.append(1*np.matmul(u,v)+0.3*Ai)
            Dlist = np.array(Dlist)
            phi =np.hstack([Dlist[i] for i in range(50)])



        from numpy.linalg import norm
        phi = phi / norm(phi,2)
        return phi
if __name__ == '__main__':
    SyntheticProblemGenerator(case=2)
