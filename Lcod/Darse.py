import logging
import numpy as np
import tensorflow as tf
from sys import stdout as out

from .utils import start_handler
from .helper_tf import soft_thresholding
from ._darseoptim_network import _DarseNetwork


class DarseNetwork(_DarseNetwork):
    """Drsae Neural Network"""
    def __init__(self,n_iters,m,n,lmbd,T,log_lvl=logging.INFO,
                 name="Darse",n_layers=2, **kwargs):
        self.n_iters = n_iters
        self.m = m
        self.n = n
        self.lmbd = lmbd
        self.T = T
        self.D = None
        self.input_Z = None


        # Logger for debugging
        self.log = logging.getLogger('DrsaeNet')
        start_handler(self.log, log_lvl)


        super(DarseNetwork,self).__init__(n_layers=n_layers, name=name, **kwargs)

    def _get_inputs(self):
        """Construct the placeholders used for the network inputs, to be passed
        as entries for the first layer.

        Return
        ------
        outputs: tuple of tensors (n_in) passed as entries to construct the 1st
                 layer of the network.
        """
        m = self.m
        n = self.n

        # with tf.name_scope("Inputs"):
        self.X = tf.placeholder(shape=[None,m], dtype=tf.float32,
                                name='X')
        # self.lmbd = tf.placeholder(dtype=tf.float32, name='lambda')

        self.Z = tf.zeros(shape=[tf.shape(self.X)[0],n], dtype=tf.float32,
                          name='Z_0')

        self.feed_map = {"Z": self.Z, "X": self.X}
        return [self.Z, self.X]

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
        Zt, X = outputs

        with tf.name_scope("reconstruction_zD"):
            # rec = tf.matmul(Zt,self.D)
            rec = Zt

        with tf.name_scope("norm_2"):
            Er = tf.multiply(
                tf.constant(.5, dtype=tf.float32),
                tf.reduce_mean(tf.reduce_sum(tf.squared_difference(X,rec),
                                             reduction_indices=[1])))

        lmbd = 0.01
        with tf.name_scope("norm_1"):
            l1 = lmbd*tf.reduce_mean(tf.reduce_sum(
                tf.abs(Zt), reduction_indices=[1]))

        return tf.add(Er, l1, name="cost")

    def _get_feed(self, batch_provider):
        """Construct the feed dictionary from the batch provider

        This method will be use to feed the network at each step of the
        optimization from the batch provider. It will put in correspondance
        the tuple return by the batch_provider and the input placeholders.
        """
        sig_batch, _, zs_batch, lmbd = batch_provider.get_batch()
        feed_dict = {self.X: sig_batch}
        if self.input_Z:
            feed_dict[self.Z] = zs_batch
        return feed_dict

    def _layer(self, inputs, params=None, id_layer=0):
        """Construct the layer id_layer in the computation graph of tensorflow.

        Parameters
        ----------
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


        Zt, X = inputs


        if self.shared and params is not None:
            self.log.debug('(Layer{}) - shared params'.format(id_layer))
            We, Ws, b ,Wd= params
        else:
            self.log.debug('(Layer{}) - new params'.format(id_layer))

            Wd = tf.Variable(tf.random_normal([self.n,self.m]),name="Wd")
            We = tf.Variable(tf.random_normal([self.m,self.n]), name='We')
            Ws = tf.Variable(tf.random_normal([self.n,self.n]),name="Ws")
            b = tf.Variable(tf.zeros([1,self.n]), name='b')
            self.D = Wd

        if id_layer==0:

            with tf.name_scope("hidden_layer"):
                hk = tf.matmul(self.X,We)
                hk += tf.matmul(Zt,Ws)
                hk -= b   # use tf.mutil ,tf.add?
            output = tf.nn.relu(hk)
        else:
            output = tf.matmul(Zt,Wd)
        tf.identity(output, name="output")

        return [output, X], (We, Ws, b, Wd)
