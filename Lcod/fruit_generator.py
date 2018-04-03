#python scripts
__author__='Du Jiawei'
#Descrption:
import numpy as np
from sklearn.utils import shuffle as shuf
import os 
import os.path as osp
import cv2
class FruitProblemGenerator():
    def __init__(self,D,lmbd,batch_size=32,source_folder="/home/dujw/darse/save_exp/fruit",seed=0):
        self.D = np.array(D)
        self.K, self.p = D.shape
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.seed = seed 
        self.source_folder = source_folder 
        self.blur_folder = source_folder + "/blur"
        self.img_list = os.listdir(self.blur_folder) 

    def get_batch(self,N=None):
        if N is None:
            N = self.batch_size
        if N > len(self.img_list) or N == -1:
            N = len(self.img_list)
        batch = shuf(self.img_list,random_state=np.random.randint(1,10000)) 
        X = []
        for i in range(N):
            X.append(self.LoadImage(osp.join(self.source_folder,batch[i])))

        X = np.array(X).reshape(N,-1)
        z = np.zeros([N,self.K])
        return X, z, z, self.lmbd
    def get_batch_with_label(self,N=None):
        if N is None:
            N = self.batch_size
        batch = shuf(self.img_list,random_state=np.random.randint(1,10000)) 
        X = []
        b = []
        for i in range(N):
            X.append(self.LoadImage(osp.join(self.source_folder,batch[i])))
            b.append(self.LoadImage(osp.join(self.source_folder,"blur",batch[i])))
        X = np.array(X).reshape(N,-1)
        b = np.array(b).reshape(N,-1)
        return X,b

    def get_test(self,N=None):
        if N is None:
            N = self.batch_size
        list = os.listdir(osp.join(self.source_folder,"test"))
        if N > len(list):
            N = len(list)
        X = []
        b = []
        for i in range(N):
            X.append(self.LoadImage(osp.join(self.source_folder,list[i])))
            b.append(self.LoadImage(osp.join(self.source_folder,"test",list[i])))
        X = np.array(X).reshape(N,-1)
        b = np.array(b).reshape(N,-1)
        return X,b

    def LoadImage(self,source):
        img = cv2.imread(source)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255
        return img

def create_dictionary_dl(lmbd, K=100, N=10000, dir_mnist='/home/dujw/darse/save_exp/fruit'):
    import os.path as osp
    D = np.random.random([32*32*3,32*32*3])
    pb = FruitProblemGenerator(D,0.1)


    fname = osp.join(dir_mnist, "D_fruit_K{}_lmbd{}.npy".format(K, lmbd))
    if osp.exists(fname):
        D = np.load(fname)
    else:
        from sklearn.decomposition import MiniBatchDictionaryLearning
        from sklearn.decomposition import MiniBatchDictionaryLearning
        X = pb.get_batch(199)[0]
        dl = MiniBatchDictionaryLearning(K, alpha=lmbd*N, fit_algorithm='cd',n_jobs=-1, verbose=1)
        dl.fit(X)
        D = dl.components_.reshape(K, -1)
        np.save(fname, D)
    return D
if __name__ == '__main__':
    D = create_dictionary_dl(0.1,1000)
