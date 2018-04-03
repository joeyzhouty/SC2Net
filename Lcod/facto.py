import numpy as np
from sys import stdout as out


class FactorizationLasso(object):
    """Realize a factorization accelerating the Lasso optimization"""
    def __init__(self, D):
        super(FactorizationLasso, self).__init__()
        self.D = D
        self.B = D.dot(D.T)

    def factorize(self, X, Z, lmbd, max_iter=2000, kappa=1.25, nu=1.25,
                  lr=1e-5, rhomom=.5, rhomomS=.5, rhomomR=.5):
        K, p = self.D.shape
        A, S, _ = np.linalg.svd(self.B)
        A = A.T

        L = max(S)
        XX = X.dot(np.linalg.pinv(self.D))
        Z1 = np.sign(XX)*np.maximum(0, abs(XX) - lmbd/K)
        disti = np.sqrt(np.mean((Z-Z1)**2, axis=1))[:, None]
        print(disti.shape)
        Znoise = Z + disti*np.random.normal(size=Z.shape)

        tatref = lmbd*np.mean(np.maximum(
            0, np.sum(abs(Z.dot(A.T))-abs(Z)-abs(Znoise.dot(A.T))+abs(Znoise),
                      axis=1)))
        Rref = L*np.eye(K) - self.B
        tatref2 = np.mean([(z-z1).T.dot(Rref).dot(z-z1)
                          for z, z1 in zip(Z, Z1)])/2

        if tatref < tatref2:
            print("Init from SVD")
        else:
            A = np.eye(K)
            S = L*np.ones(K)
            print('init from identity')

        trt, tat, self.train_cost = [], [], []
        # mom, momS = 0, 0
        momR, momX = 0, 0
        rhomomX = rhomomR
        Rt = A.T.dot(np.diag(S)).dot(A) - self.B
        Xt = np.zeros(Rt.shape)
        dR1 = (Z-Z1).T.dot(Z-Z1)
        dR = np.copy(dR1)
        dX = np.zeros(dR1.shape)

        Ax, Sx = np.copy(A), np.copy(S)

        delta_a = DeltaA(lmbd, Z, Znoise)
        # delta_a._check_gradient(A)

        for it in range(1, max_iter+1):
            if it % max_iter//4 == 0:
                lr *= .9

            # Rt = A.T.dot(np.diag(S)).dot(A) - self.B
            Znoise = Z + disti*np.random.normal(size=Z.shape)

            cA, dA = delta_a.grad(A)
            # cAx, dAx = delta_a.grad(Ax)
            # tmp, dS = self._get_dS(A, S, Rt, Z-Z1, nu)
            # dA += tmp

            # # Project on unitary matrix space
            dA = dA - A.dot(dA.T).dot(A)

            # trt += [np.mean([np.sum((Xt.dot(z))**2) for z in Z-Z1])/2]
            trt += [np.mean([z.dot(Rt.dot(z)) for z in Z-Z1])/2]
            tat += [cA]
            # tet += [kappa*np.sum(tutu*tutu)]
            self.train_cost += [trt[-1]+tat[-1]]  # +tet[-1]]
            # mom = -lr*dA + rhomom*mom
            # momS = -lr*dS + rhomomS*momS

            dR[:, :] = dR1  # np.mean([np.outer(z, z) for z in (Z-Z1)], axis=0)

            # dX[:, :] = np.mean([Xt.dot(np.outer(z, z))
            #                     for z in (Z-Z1)], axis=0)
            dR2 = (1/np.maximum(S[:, None], 1e-6)*A.T).dot(dA)
            # print('\n{:.4e} {:.4e}'.format(
            #     np.sum((dR2+dR2.T)**2), np.sum(dR1*dR1)))
            dR += dR2+dR2.T
            # dX2 = (1/np.sqrt(np.maximum(Sx[:, None], 1e-12))*Ax.T).dot(dAx)
            # dX += dX2+dX2.T
            assert np.allclose(dR, dR.T)
            momR = -lr*dR + rhomomR*momR
            # momX = -lr*dX + rhomomX*momX
            # A += mom
            # S += momS
            Rt += momR
            Xt += momX
            # P, s, Q = np.linalg.svd(Rt+Rt.T)
            # Rt = P.dot(np.maximum(0, s)[:, None]*Q)/2
            # e, v = np.linalg.eig(Rt)
            # I = e > 0
            # Rt = np.real(v[:, I].dot((e[I, None]*v[:, I].conj().T)))
            A, S, _ = np.linalg.svd(Rt + self.B)
            # Rt = A.T.dot(S*A) - self.B
            # Ax, Sx, _ = np.linalg.svd(Xt.T.dot(Xt) + self.B)

            # P, _, Q = np.linalg.svd(A)
            # A = P.dot(Q)

            if it % 50 == 0 or it == 1:
                print('\rFactorize {:7.2%}: {:8.6f}, {:7.5e}, {:8.6f}, {:7.5e}'
                      .format(it/max_iter, self.train_cost[-1], trt[-1],
                              tat[-1], min(S)))
            out.write('\rFactorize {:7.2%}: {:8.6f}, {:7.5e}, {:8.6f}, {:7.5e}'
                      .format(it/max_iter, self.train_cost[-1], trt[-1],
                              tat[-1], min(S)))
            out.flush()
        print('\rFactorize {:7}: {:8.6f}, {:7.5e}, {:8.6f}, {:7.5e}'.format(
            'done', self.train_cost[-1], trt[-1], tat[-1], min(S)))

        return A, S

    def _get_dA(self, A, Z, Z1, lmbd):
        t0 = np.mean(abs(Z.dot(A.T))-abs(Z)-abs(Z1.dot(A.T))+abs(Z1), axis=1)
        I0 = t0 > 0

        T1 = Z.T.dot(I0[:, None]*np.sign(Z.dot(A.T)))
        T2 = Z1.T.dot(I0[:, None]*np.sign(Z1.dot(A.T)))
        t1 = np.mean([np.outer(z, i0*np.sign(z.dot(A.T)))
                      for z, i0 in zip(Z, I0)], axis=0)
        t2 = np.mean([np.outer(z, i0*np.sign(z.dot(A.T)))
                      for z, i0 in zip(Z1, I0)], axis=0)
        t1, t2
        return lmbd*(T1 - T2)/Z.shape[0]

    def _get_dS(self, A, S, R, Z, nu):
        dR = self._get_dR(R, Z, nu)
        dA = 2 * dR.dot(A.T)*S[None, :]
        dS = A.dot(dR).dot(A.T)
        dS = np.diag(dS)
        return dA, dS

    def _get_dR(self, R, Z, nu):
        # dR1 = np.sum([np.outer(z, z) for z in Z], axis=0)
        dR1 = Z.T.dot(Z)
        dd, vv = np.linalg.eig(R)
        II = dd < 0
        dR2 = -nu*np.real(vv[:, II].dot(vv[:, II].T))
        return dR1 + dR2


class DeltaA(object):
    """Op to compute and derive delta A"""
    def __init__(self, lmbd, Z, Znoise):
        super(DeltaA, self).__init__()
        self.lmbd = lmbd
        self.Z = Z
        self.Znoise = Znoise

    def __call__(self, A):
        return self.lmbd*np.mean(np.maximum(np.sum(
                abs(self.Z.dot(A.T)) - abs(self.Z) +
                abs(self.Znoise) - abs(self.Znoise.dot(A.T)), axis=1), 0))

    def grad(self, A):
        t0 = np.sum(abs(self.Z.dot(A.T))-abs(self.Z) -
                    abs(self.Znoise.dot(A.T))+abs(self.Znoise), axis=1)
        I0 = t0 > 0

        if np.any(I0):
            t1 = (np.sign(self.Z[I0].dot(A.T))).T.dot(self.Z[I0])
            t2 = (np.sign(self.Znoise[I0].dot(A.T))).T.dot(self.Znoise[I0])
            # t1 = np.sum([np.outer(i0*np.sign(A.dot(z)), z)
            #              for z, i0 in zip(self.Z, I0) if i0], axis=0)
            # t2 = np.sum([np.outer(i0*np.sign(A.dot(z)), z)
            #              for z, i0 in zip(self.Znoise, I0) if i0], axis=0)
        else:
            t1, t2 = np.zeros(A.shape), 0
        # t1 = np.mean([np.outer(i0*np.sign(A.dot(z)), z)
        #               for z, i0 in zip(self.Z, I0)], axis=0)
        # t2 = np.mean([np.outer(i0*np.sign(A.dot(z)), z)
        #               for z, i0 in zip(self.Znoise, I0)], axis=0)
        cA = self.lmbd*np.mean(np.maximum(t0, 0))
        return cA, self.lmbd*(t1 - t2)

    def _check_gradient(self, A, n_checks=100):
        t = []
        for k in range(n_checks):
            A0 = np.random.normal(size=A.shape)
            c0, gA = self.grad(A0)
            c3 = self(A0 - gA)

            # # Project on unitary matrix space
            # gA = gA.dot(A.T) - A.dot(gA.T)
            # e, v = np.linalg.eig(gA)
            # P = v.dot(np.exp(-4*e)[:, None]*v.T)
            # c4 = self(P.dot(A0))
            # assert c0 >= c4, "No unit descent: {} < {}".format(c0, c4)

            t += [abs(c0-c3)]
            out.write("\rChecking gradient: {:7.2%} - {}, {}, {}".format(
                k/n_checks, c0, c3, c0-c3))
            assert c0 >= c3, "No descent: {} < {}, {}".format(
                c0, c3, np.mean(t))
            eps = 1e-4*np.random.normal(size=A.shape)
            nH = np.sum(eps*eps)
            c1 = self(A0+eps)
            c2 = c0 + np.sum(gA*eps)
            dc = abs(c1 - c2)
            out.write("\rChecking gradient: {:7.2%} - {}, {}, {}".format(
                k/n_checks, dc, nH, c0))
            out.flush()
            assert np.isclose(c1, c2, atol=np.sqrt(nH))
        print("\rChecking gradient: {:7} - {}".format('ok', np.mean(t)))
