from __future__ import division
import sys 
sys.path.append("/home/dujw/darse/data_handlers/recon_sparse")
from evaluate import cal_acry
import numpy as np
import tensorflow as tf
from sys import stdout as out
from sklearn.linear_model import LogisticRegression as lgr
from sklearn.metrics import accuracy_score



class _OptimTF(object):
    """Iterative Soft thresholding algorithm in TF"""
    def __init__(self, name, gpu_usage=.9):
        super(_OptimTF,self).__init__()
        self.name = name
        self.gpu_usage = gpu_usage
        self._construct()
        self.reset()

    def test_accuracy(self,pb,lmbd=0.1,max_iter=100):

        test_x,test_y = pb.get_batch_with_label(8000)
        test_y = np.argmax(test_y,axis=1)
        feed_final = {"X": test_x[:6000], "lmbd": lmbd}
        train_data = self.optimize(X=test_x[:6000],lmbd=lmbd,max_iter=max_iter) 

        lgrs = lgr()
        lgrs.fit(train_data,test_y[:6000])

        # test_x,test_y = pb.get_batch_with_label(1000)
        # test_y = np.argmax(test_y,axis=1)
        lis_out = self.optimize(X=test_x[6000:],lmbd=lmbd,max_iter=max_iter) 
        y_pre = lgrs.predict(lis_out)
        return accuracy_score(test_y[6000:],y_pre)

    def test_loss(self,pb,lmbd=0.1,max_iter=100):
        test_x,test_y = pb.get_batch_with_label(8000)
        feed_final = {"X": test_x, "lmbd": lmbd}
        train_data = self.optimize(X=test_x,lmbd=lmbd,max_iter=max_iter) 
        from numpy.linalg import norm 
        sum = 0.0
        # for i in range(8000):
            # sum+=norm(train_data[i]-test_y[i],2)
        # return sum/8000
        for i in range(8000):
            sum += cal_acry(train_data[i],test_y[i])
        return sum/8000
    def reconstruct(self,pb,lmbd=0.1,max_iter=100):
        test_x  = pb.get_batch(100,shuffle=False)[0]
        train_data = self.optimize(X=test_x,lmbd=lmbd,max_iter=max_iter) 
        recu_x = []
        for i in train_data:
            rec = np.matmul(i,self.D)
            recu_x.append(rec)
        recu_x = np.array(recu_x)
        from utils import drow_mnist
        from utils import drow_cifar
        if test_x.shape[1]>1000:
            drow_cifar(test_x,recu_x)
        else:
            drow_mnist(test_x,recu_x)



    def _construct(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = self._get_inputs()
            self.step_optim, self.dz = self._get_step(self.inputs)
            self._cost = self._get_cost(self.inputs)
            self.var_init = tf.initialize_all_variables()

    def reset(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_usage
        self.session = tf.Session(graph=self.graph, config=config)
        self.train_cost = []
        self.session.run(self.var_init)

    def output(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        return self._output.eval(feed_dict=feed_dict, session=self.session)

    def cost(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        return self.session.run(self._cost, feed_dict=feed_dict)

    def optimize(self, X, lmbd, Z=None, max_iter=1, tol=1e-5):
        if Z is None:
            batch_size = X.shape[0]
            K = self.D.shape[0]
            z_curr = np.zeros((batch_size, K))
        else:
            z_curr = np.copy(Z)
        self.train_cost = []
        feed = {self.X: X, self.Z: z_curr, self.lmbd: lmbd}
        for k in range(max_iter):
            z_curr[:], dz, cost = self.session.run(
                [self.step_optim, self.dz, self._cost], feed_dict=feed)
            self.train_cost += [cost]
            if dz < tol:
                print("\r{} reached optimal solution in {}-iteration"
                      .format(self.name, k))
                break
            out.write("\rIterative optimization ({}): {:7.1%} - {:.4e}"
                      "".format(self.name, k/max_iter, dz))
            out.flush()
        self.train_cost += [self.session.run(self._cost, feed_dict=feed)]
        print("\rIterative optimization ({}): {:7}".format(self.name, "done"))
        return z_curr

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            _feed[self.feed_map[k]] = v
        return _feed

    def terminate(self):
        self.session.close()
