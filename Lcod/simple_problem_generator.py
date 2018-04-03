import numpy as np


class SimpleProblemGenerator(object):
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
    def __init__(self, D, lmbd, rho=None, n_block=1,
                 corr=None, batch_size=100, seed=None):
        super(SimpleProblemGenerator, self).__init__()
        self.D = np.array(D)
        self.K, self.p = D.shape
        if rho is None:
            rho = 5/self.K
        self.rho = rho
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        if corr is None:
            corr = 0
        self.corr, self.t = self._generate_block_binary(n_block, corr)
        self.sig = (1-corr)*np.eye(self.K) + corr*(self.corr > 0)

    def get_batch(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size
        K = self.K
        # Draw a bernouilli with parameter 1-rho
        J0 = self.rng.multivariate_normal([0]*K, self.corr, N) > self.t
        # Draw coeffcients from a normal law $\mathcal N(0, 1)$
        z = 10*self.rng.multivariate_normal([0]*K, self.sig, N)
        z = z.astype(np.float32)*J0

        # Draw a random starting point for the problem
        # z_start = self.rng.random_sample(size=(N, z.shape[-1]))
        z_start = np.zeros(z.shape, dtype=np.float32)
        sig = z.dot(self.D)
        return sig, z, z_start, self.lmbd

    def get_test(self, N):
        return self.get_batch(N)

    def lasso_cost(self, z, sig):
        '''Cost of the point z for a problem with sig'''
        residual = sig - z.dot(self.D)
        Er = np.sum(residual*residual, axis=1)/2
        return Er.mean() + self.lmbd*abs(z).sum(axis=1).mean()

    def lasso_std(self, z, z_0, sig):
        '''Cost of the point z for a problem with sig'''
        res = sig - z.dot(self.D)
        Er = np.sum(res*res, axis=1)/2 + self.lmbd*abs(z).sum(axis=1)
        res = sig - z_0.dot(self.D)
        Er_0 = np.sum(res*res, axis=1)/2 + self.lmbd*abs(z).sum(axis=1)
        return np.std(Er-Er_0)

    def _generate_block_binary(self, k, corr=.9):
        '''Generate multivariate bernouilli with parameter rho
        and non null block correlation
        '''
        from math import sqrt
        from scipy.special import erfcinv
        sig = np.eye(self.K)
        sb = self.K // k

        # P(X_i = 1)= rho
        t = sqrt(2)*erfcinv(2*self.rho)

        for i in range(k):
            ss = sig[i*sb:(i+1)*sb, i*sb:(i+1)*sb].shape[0]
            sig[i*sb:(i+1)*sb, i*sb:(i+1)*sb] = ((1-corr)*np.eye(ss) +
                                                 corr*np.ones((ss, ss)))
        return sig, t


def create_dictionary(K, p, seed=None):
    np.random.seed(seed)
    D = np.random.normal(size=(K, p)).astype(np.float32)
    D /= (D*D).sum(axis=1)[:, None]
    D = D.astype(np.float32)
    return D
