import logging
import numpy as np
import tensorflow as tf
from sys import stdout as out

from .utils import start_handler
from .helper_tf import soft_thresholding
from ._loptim_network import _LOptimNetwork
from .ista_tf import IstaTF
from sklearn.linear_model import LogisticRegression as lgr
from sklearn.metrics import accuracy_score


class LIstaNetwork(_LOptimNetwork):
    """Lifsta Neural Network"""
    def __init__(self, D, n_layers, input_Z=False, log_lvl=logging.INFO,
                 name=None, Zpflag=False,feed_lmbd = 0.1,**kwargs):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2
        self.Zpflag = Zpflag
        self.ista=IstaTF(D,gpu_usage=100)
        self.feed_lmbd=feed_lmbd

        # Option for the network
        self.input_Z = input_Z

        # Logger for debugging
        self.log = logging.getLogger('LIstaNet')
        start_handler(self.log, log_lvl)

        if name is None:
            name = 'L-ISTA_{:03}'.format(n_layers)

        super(LIstaNetwork,self).__init__(n_layers=n_layers, name=name, **kwargs)

#     def test_accuracy(self,pb,lmbd=0.1):

        # test_x,test_y = pb.get_batch_with_label(4000)
        # test_y = np.argmax(test_y,axis=1)
        # zs_test = np.zeros((4000,self.D.shape[0]))
        # feed_final = {"Z": zs_test[:4000], "X": test_x, "lmbd": lmbd}
        # train_data = self.output(**feed_final)

        # lgrs = lgr()
        # lgrs.fit(train_data,test_y)

        # test_x,test_y = pb.get_batch_with_label(1000)
        # test_y = np.argmax(test_y,axis=1)
        # feed_final = {"Z": zs_test[:1000], "X": test_x, "lmbd": lmbd}
        # lis_out = self.output(**feed_final)
        # y_pre = lgrs.predict(lis_out)
        # return accuracy_score(test_y,y_pre)



    def _get_inputs(self):
        """Construct the placeholders used for the network inputs, to be passed
        as entries for the first layer.

        Return
        ------
        outputs: tuple of tensors (n_in) passed as entries to construct the 1st
                 layer of the network.
        """
        K, p = self.D.shape
        # with tf.name_scope("Inputs"):
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,
                                name='X')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lambda')

        self.label = tf.placeholder(shape=[None,10],dtype=tf.float32,name="label")
        self.Z = tf.zeros(shape=[tf.shape(self.X)[0], K], dtype=tf.float32,
                          name='Z_0')
        self.Zp = tf.zeros(shape=[tf.shape(self.X)[0], K], dtype=tf.float32,
                          name='Z_p')


        self.feed_map = {"Z": self.Z, "Zp":self.Zp,"X": self.X, "label":self.label,"lmbd": self.lmbd}
        return [self.Z, self.Zp,self.X, self.label,self.lmbd]

    def _get_cost(self, outputs):
        """Construct the cost function from the outputs of the last layer. This
        will be used through SGD to train the network.

        Parameters
        ----------
        outputs: tuple fo tensors (n_out)
            a tuple of tensor containing the output from the last layer of the
            network

        Returns
        -------
        cost: a tensor computing the cost function of the network.
        reg: a tensor for computing regularisation of the parameters.
            It should be None if no regularization is needed.
        """
        Zk, Zp, X, label,lmbd = outputs
        L = self.L

        #get the classify matrix 
        if self.supervised == True:
            wc = tf.trainable_variables()[0]
            wd = L * tf.trainable_variables()[2]
            self.CC = wc
        else:
            wd = L * tf.trainable_variables()[1]
        wd = tf.transpose(wd)
        self.DD = wd


        if self.Zpflag == False:

            with tf.name_scope("reconstruction_zD"):
                # rec = tf.matmul(Zk, tf.constant(self.D))
                rec = tf.matmul(Zk,wd) 

            with tf.name_scope("norm_2"):
                Er = tf.multiply(
                    tf.constant(.5, dtype=tf.float32),
                    tf.reduce_mean(tf.reduce_sum(tf.squared_difference(rec, X),
                                                 reduction_indices=[1])))

            with tf.name_scope("norm_1"):
                l1 = lmbd*tf.reduce_mean(tf.reduce_sum(
                    tf.abs(Zk), reduction_indices=[1]))

            if self.supervised==True:
                with tf.name_scope("class_y"):
                    # ly = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(tf.nn.softmax(tf.matmul(Zk,wc)),label),reduction_indices=[1]))
                    ly = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(Zk,wc),labels=label)))
                    ly = 0.1* ly
                l1 = tf.add(l1,ly)
                return tf.add(Er,l1,name="cost")
            else:
                return tf.add(Er, l1,name="cost")
        else: 
            Cost = tf.multiply(tf.constant(.5,dtype=tf.float32),tf.reduce_mean(tf.reduce_sum(tf.squared_difference(Zp, Zk),reduction_indices=[1])))
            return Cost

    def _get_feed(self, batch_provider):
        """Construct the feed dictionary from the batch provider

        This method will be use to feed the network at each step of the
        optimization from the batch provider. It will put in correspondance
        the tuple return by the batch_provider and the input placeholders.
        """
        sig_batch, _, zs_batch, lab,lmbd = batch_provider.get_batch()
        lab = lab.astype(np.float32)
        feed_dict = {self.X: sig_batch,self.label:lab,
                     self.lmbd: lmbd}
        if self.Zpflag == True:
            feed_dict[self.Zp] = self.ista.optimize(X=sig_batch,lmbd=self.feed_lmbd,Z=zs_batch,max_iter=1000)
        if self.input_Z:
            feed_dict[self.Z] = zs_batch
        return feed_dict

    def _layer(self, inputs, params=None, id_layer=0):
        """Construct the layer id_layer in the computation graph of tensorflow.

        Parameters
        ---------
        inputs: tuple of tensors (n_in)
            a tuple of tensor containing all the necessary inputs to construct
            the layer, either network inputs or previous layer output.
        params: tuple of tensor (n_param)
            a tuple with the parameter of the previous layers, used to share
            the parameters accross layers. This is not used if the network do
            not use the shared parameter.
        id_layer: int
            A layer identifier passed during the construction of the network.
            It should be its rank in the graph.
        Returns
        -------
        outputs: tuple of tensors (n_out) st n_out = n_in, to chain the layers.
        params: tuple of tensors (n_param) with the parameters of this layer
        """
        L = self.L
        K, p = self.D.shape
        Zk, Zp, X, label,lmbd = inputs
        if self.shared and params is not None:
            self.log.debug('(Layer{}) - shared params'.format(id_layer))
            Wg, We, theta = params
        else:
            if len(self.warm_param) > id_layer:
                self.log.debug('(Layer{})- warm params'.format(id_layer))
                wp = self.warm_param[id_layer]
            else:
                self.log.debug('(Layer{}) - new params'.format(id_layer))
                wp = [np.eye(K, dtype=np.float32) - self.S0/L,
                      (self.D.T).astype(np.float32)/L,
                      np.ones(K, dtype=np.float32)/L]

            if id_layer > 0 or self.input_Z or self.shared:
                Wg = tf.Variable(tf.constant(wp[0], shape=[K, K]), name='Wg')
            else:
                Wg = None
            We = tf.Variable(tf.constant(wp[1], shape=[p, K]), name='We')
            theta = tf.Variable(tf.constant(wp[2], shape=[K]), name='theta')

        with tf.name_scope("hidden_layer"):
            hk = tf.matmul(self.X, We)
            if id_layer > 0 or self.input_Z:
                hk += tf.matmul(Zk, Wg)
        output = soft_thresholding(hk, self.lmbd*theta)
        tf.identity(output, name="output")

        return [output, Zp, X, label, lmbd], (Wg, We, theta)
