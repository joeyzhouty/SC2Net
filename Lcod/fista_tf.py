import numpy as np
import tensorflow as tf
from sys import stdout as out

from .helper_tf import soft_thresholding
from ._optim_tf import _OptimTF


class FistaTF(_OptimTF):
    """Iterative Soft thresholding algorithm in TF"""
    def __init__(self, D, name=None, gpu_usage=.9):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2

        super(FistaTF,self).__init__(name=name if name else 'Fista', gpu_usage=gpu_usage)

    def _get_inputs(self):
        K, p = self.D.shape
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,  name='X')
        self.Z = tf.zeros(shape=[tf.shape(self.X)[0], K], dtype=tf.float32,
                          name='Zk')
        self.Y = tf.zeros_like(self.Z, name='Yk')
        self.theta = tf.placeholder(dtype=tf.float32, name='theta')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lmbd')
        self.feed_map = {"Z": self.Y, "X": self.X, "theta": self.theta,
                         "lmbd": self.lmbd}

        return (self.Z, self.Y, self.X, self.theta, self.lmbd)

    def _get_step(self, inputs):
        Z, Y, X, theta, lmbd = self.inputs
        K, p = self.D.shape
        L = self.L
        with tf.name_scope("ISTA_iteration"):
            self.S = tf.constant(np.eye(K, dtype=np.float32) - self.S0/L,
                                 shape=[K, K], name='S')
            self.We = tf.constant(self.D.T/L, shape=[p, K],
                                  dtype=tf.float32, name='We')
            hk = tf.matmul(Y, self.S) + tf.matmul(X, self.We)
            self.step_FISTA = Zk = soft_thresholding(hk, lmbd/L)
            # self.theta_k = tk = (tf.sqrt(theta*theta+4) - theta)*theta/2
            self.theta_k = tk = (1 + tf.sqrt(1 + 4*theta*theta))/2
            dZ = tf.subtract(Zk, Z)
            # self.Yk = Zk + tk*(1/theta-1)*dZ
            self.Yk = Zk + (theta-1)/tk*dZ
            self.dz = tf.reduce_mean(tf.reduce_sum(
                dZ*dZ, reduction_indices=[1]))

            step = tf.tuple([Zk, tk, self.Yk])
        return step, self.dz

    def _get_cost(self, inputs):
        Z, _, X, _, lmbd = self.inputs
        with tf.name_scope("Cost"):
            rec = tf.matmul(Z, tf.constant(self.D))
            Er = tf.reduce_mean(
                tf.reduce_sum(tf.squared_difference(rec, X),
                              reduction_indices=[1]))/2
            cost = Er + lmbd*tf.reduce_mean(
                tf.reduce_sum(tf.abs(Z), reduction_indices=[1]))

        return cost

    def optimize(self, X, lmbd, Z=None, max_iter=1, tol=1e-5):
        if Z is None:
            batch_size = X.shape[0]
            K = self.D.shape[0]
            y_curr = np.zeros((batch_size, K))
        else:
            y_curr = np.copy(Z)

        z_curr = np.copy(y_curr)
        feed = {self.X: X, self.Z: z_curr, self.Y: y_curr,
                self.theta: 1, self.lmbd: lmbd}
        feed2 = {self.X: X, self.Z: y_curr, self.lmbd: lmbd}
        self.train_cost = []
        for k in range(max_iter):
            z_curr[:], y_curr[:], tk, dz, cost = self.session.run([
                self.step_FISTA, self.Yk, self.theta_k, self.dz, self._cost,
                ], feed_dict=feed)
            feed[self.theta] = tk
            self.train_cost += [cost]
            if dz < tol:
                print("\r{} reached optimal solution in {}-iteration"
                      .format(self.name, k+1))
                break
            out.write("\rIterative optimization ({}): {:7.1%} - {:.4e}"
                      "".format(self.name, k/max_iter, dz))
            out.flush()
        self.train_cost += [self.session.run(self._cost, feed)]
        print("\rIterative optimization ({}): {:7}".format(self.name, "done"))
        return z_curr
