
import sys
import numpy as np
from glob import glob
import os.path as osp
from scipy.misc import imread

DATA_URL = ("http://host.robots.ox.ac.uk/pascal/VOC/voc2008/"
            "VOCtrainval_14-Jul-2008.tar")
DATA_DIR = osp.join("save_exp","images","VOC")
TAR_FILE = osp.join(DATA_DIR, "VOC.tar")


def report(i, k, K):
    sys.stdout.write("\rLoad Pacal Images: {:7.2%}".format(i*k/K))
    sys.stdout.flush()


def get_members(tar):
    for member in tar.getmembers():
        if 'JPEG' in member.name:
            member.name = osp.basename(member.name)
            yield member


class ImageProblemGenerator(object):
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
    def __init__(self, D, lmbd, batch_size=100, data_dir=DATA_DIR, seed=None):
        super(ImageProblemGenerator,self).__init__()
        self.D = np.array(D)
        self.K, self.p = D.shape
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.patch_size = ps = int(np.sqrt(self.p))
        import pdb
        pdb.set_trace()

        # Load training images
        fnames = glob(osp.join(data_dir, '*jpg'))
        if len(fnames) == 0:
            # Load the images form VOC2008 and extract the archive
            import urllib
            import tarfile
            urllib.urlretrieve(DATA_URL, TAR_FILE, reporthook=report)
            tar = tarfile.open(TAR_FILE)
            tar.extractall(path=DATA_DIR, members=get_members(tar))

        self.rng.shuffle(fnames)
        print("Found {} training images".format(len(fnames)))
        f_im_train = fnames[:500]
        self.im = [imread(fn, mode='L')/255 for fn in f_im_train]
        self.im = [im - .5 for im in self.im]
        self.patches = [(im.shape[0]-ps)*(im.shape[1]-ps)
                        for im in self.im]
        self.n_patches = np.sum(self.patches)
        self.N_im = len(self.im)

        # Load test images
        f_im_test = fnames[500:600]
        self.im_test = [imread(fn, mode='L')/255 for fn in f_im_test]
        self.im_test = [im - .5 for im in self.im_test]
        self.patches_test = [(im.shape[0]-ps)*(im.shape[1]-ps)
                             for im in self.im_test]
        self.n_patches_test = np.sum(self.patches_test)
        self.N_im_test = len(self.im_test)

    def get_batch(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size
        ps = self.patch_size

        # Draw a bernouilli with parameter 1-rho
        I0 = self.rng.randint(low=0, high=self.N_im, size=N)
        J0 = self.rng.rand(N)
        X = []
        for i, j in zip(I0, J0):
            im, n_p = self.im[i], self.patches[i]
            w = (int(j*n_p) % (im.shape[0]-ps))
            h = (int(j*n_p) // (im.shape[0]-ps))
            X += [self.im[i][w:w+ps, h:h+ps].flatten()]
            X[-1] -= X[-1].mean()
        z = np.zeros((N, self.K))
        X = np.array(X)
        return X, z, z, self.lmbd

    def get_test(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size
        ps = self.patch_size

        # Draw a bernouilli with parameter 1-rho
        I0 = self.rng.randint(low=0, high=self.N_im_test, size=N)
        J0 = self.rng.rand(N)
        X = []
        for i, j in zip(I0, J0):
            im, n_p = self.im_test[i], self.patches_test[i]
            w = (int(j*n_p) % (im.shape[0]-ps))
            h = (int(j*n_p) // (im.shape[0]-ps))
            X += [self.im_test[i][w:w+ps, h:h+ps].reshape(-1)]
            X[-1] -= X[-1].mean()
        z = np.zeros((N, self.K))
        return X, z, z, self.lmbd

    def lasso_cost(self, z, sig):
        '''Cost of the point z for a problem with sig'''
        residual = sig - z.dot(self.D)
        Er = np.sum(residual*residual, axis=1)/2
        return Er.mean() + self.lmbd*abs(z).sum(axis=1).mean()


def create_dictionary_haar(p=8):
    import pywt
    c = pywt.wavedec2(np.zeros((p, p)), 'haar')
    D = []
    for k in range(1, len(c)):
        for i in range(3):
            ck = c[k][i]
            l = ck.shape[0]
            for j in range(l):
                for m in range(l):
                    ck[j, m] = 1
                    D += [pywt.waverec2(c, 'haar')]
                    ck[j, m] = 0
    ck = c[0]
    l = ck.shape[0]
    for j in range(l):
        for m in range(l):
            ck[j, m] = 1
            D += [pywt.waverec2(c, 'haar')]
            ck[j, m] = 0
    D = np.array(D).reshape(-1, p*p)

    Dn = []
    for i in range(15):
        Dn += translate(D[i].reshape((p, p)))
    return np.array(Dn).reshape((-1, p*p))


def translate(d):
    D1 = []
    for u in range(-4, 5, 2):
        for v in range(-4, 5, 2):
            d1 = np.zeros((8, 8))
            ws1, we1 = max(0, u), u if u < 0 else None
            ws, we = (-u, None) if u <= 0 else (None, -u)
            hs1, he1 = (0, v) if v < 0 else (v, None)
            hs, he = (-v, None) if v <= 0 else (None, -v)
            d1[ws1:we1, hs1:he1] = d[ws:we, hs:he]
            D1 += [d1]
    return D1
