from __future__ import division
import logging
import numpy as np
import tensorflow as tf

from .utils import start_handler
from .helper_tf import soft_thresholding
from ._loptim_network import _LOptimNetwork


class FactoNetwork(_LOptimNetwork):
    """Lifsta Neural Network"""
    def __init__(self, D, n_layers, reg_unary=True, run_svd=False,
                 proj_A=False, manifold=False, log_lvl=logging.INFO, name=None,
                 sgd=False, **kwargs):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2
        self.reg_unary = reg_unary
        self.manifold = manifold
        self.run_svd = run_svd
        self.proj_A = proj_A
        self.sgd = sgd

        self.log = logging.getLogger('FactoNet')
        start_handler(self.log, log_lvl)

        if name is None:
            name = 'FacNet_{:03}'.format(n_layers)

        super(FactoNetwork,self).__init__(n_layers=n_layers, name=name, **kwargs)

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

        self.Z = tf.zeros(shape=[tf.shape(self.X)[0], K], dtype=tf.float32,
                          name='Z_0')
        self.reg_unitary = tf.constant(0, dtype=tf.float32)

        self.feed_map = {"Z": self.Z, "X": self.X, "lmbd": self.lmbd}
        return [self.Z, self.X, self.lmbd]

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
        cost: a tensor computing the cost function of the network
        reg: a tensor for computing regularisation of the parameters.
            It should be 0 if no regularization is needed.
        """
        Zk, X, lmbd = outputs

        with tf.name_scope("reconstruction_zD"):
            rec = tf.matmul(Zk, tf.constant(self.D))

        with tf.name_scope("norm_2"):
            Er = .5*tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(rec, X), reduction_indices=[1]))

        with tf.name_scope("norm_1"):
            l1 = lmbd*tf.reduce_mean(tf.reduce_sum(
                tf.abs(Zk), reduction_indices=[1]))

        cost = tf.add(Er, l1, name="cost")
        return cost

    def _get_feed(self, batch_provider):
        """Construct the feed dictionary from the batch provider

        This method will be use to feed the network at each step of the
        optimization from the batch provider. It will put in correspondance
        the tuple return by the batch_provider and the input placeholders.
        """
        sig_batch, _, zs_batch, lmbd = batch_provider.get_batch()
        feed_dict = {self.Z: zs_batch,
                     self.X: sig_batch,
                     self.lmbd: lmbd}
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
        L = self.L
        K, p = self.D.shape
        Zk, X, lmbd = inputs
        D = tf.constant(self.D)
        DD = tf.constant(self.S0)
        if params:
            self.log.debug('(Layer{}) - shared params'.format(id_layer))
            A, S = params
        else:
            if len(self.warm_param) > id_layer:
                self.log.debug('(Layer{})- warm params'.format(id_layer))
                wp = self.warm_param[id_layer]
            else:
                self.log.debug('(Layer{}) - new params'.format(id_layer))
                wp = [np.eye(K, dtype=np.float32),
                      np.ones(K, dtype=np.float32)*L]
            A = tf.Variable(initial_value=tf.constant(wp[0], shape=[K, K]),
                            name='A')
            S = tf.Variable(tf.constant(wp[1], shape=[K]), name='S')

            # Projection of A on the stieffel manifold
            with tf.name_scope("unary_projection"):
                _, P, Q = tf.svd(tf.cast(A, tf.float64),
                                 full_matrices=True)
                An = tf.matmul(P, Q, transpose_b=True)
                tf.add_to_collection('svd', A.assign(tf.cast(An, tf.float32)))
            tf.add_to_collection('Unitary', A)

            with tf.name_scope('unit_reg'):
                I = tf.constant(np.eye(K, dtype=np.float32))
                r = tf.squared_difference(I, tf.matmul(A, A,
                                          transpose_a=True))
                tf.add_to_collection("regularisation", tf.reduce_sum(r))
        S1 = 1 / S
        as1 = tf.matmul(A, tf.diag(S1))

        with tf.name_scope("hidden"):
            hk = tf.matmul(self.X, tf.matmul(D, as1, transpose_a=True))
            if id_layer > 0:
                hk += tf.matmul(Zk, (A-tf.matmul(DD, as1)))
        output = soft_thresholding(hk, self.lmbd*S1)
        output = tf.matmul(output, A, transpose_b=True, name="output")

        return [output, X, lmbd], (A, S)

    def _mk_training_step(self):
        """Function to construct the training steps and procedure.

        This function returns an operation to iterate and train the network.
        """
        # Training methods
        if not self.sgd:
            self._optimizer = tf.train.AdagradOptimizer(
                self.lr, initial_accumulator_value=self.init_value_ada)
        else:
            self._optimizer = tf.train.GradientDescentOptimizer(
                self.lr)

        if not self.reg_unary:
            _reg = tf.constant(0, dtype=tf.float32)
        else:
            _reg = tf.add_n(tf.get_collection_ref("regularisation"))
            _reg /= self.n_layers

        # For parameters A of the layers, use Adagrad in the Stiefel manifold
        grads_and_vars = self._optimizer.compute_gradients(
            self._cost+self.reg_scale*_reg)
        for i, (g, v) in enumerate(grads_and_vars):
            if v in tf.get_collection('Unitary'):
                if self.proj_A:
                    # gA = gA - A.gA^T.A
                    g = g - tf.matmul(v, tf.matmul(g, v, transpose_a=True))
                    grads_and_vars[i] = (g, v)
        _train = self._optimizer.apply_gradients(
            grads_and_vars, global_step=self.global_step)

        _svd = tf.get_collection('svd')
        if self.manifold:
            # Use the dependency to project only once Adagrad has done its step
            with tf.control_dependencies([_train]):
                _train = tf.group(*_svd)
        else:
            self._svd = tf.group(*_svd)
            s1 = tf.summary.scalar("cost (pre svd)",
                                   self._cost-self.feed_map['c_val'])

            # summary to track manifold deviation
            s2 = tf.summary.scalar('cost_manifold', tf.add_n(
                tf.get_collection("regularisation")))
            self._pre_svd = tf.summary.merge([s1, s2])
        return _train

    def epoch(self, lr_init, reg_cost, tol):

        if not self.manifold and self.run_svd:
            it = self.global_step.eval()
            summary = self.session.run(self._pre_svd, feed_dict=self._feed_val)
            self.writer.add_summary(summary, global_step=it)
            self._svd.run()

        return super(FactoNetwork,self).epoch(lr_init, reg_cost, tol)

    def train(self, batch_provider, feed_val, max_iter, steps, lr_init=.01,
              tol=1e-5, reg_cost=15, model_name='loptim', save_model=False):
        """Train the network
        """
        from sys import stdout as out
        self._feed_val = self._convert_feed(feed_val)
        self._last_downscale = -reg_cost
        training_cost = 1e100
        with self.session.as_default():
            for k in range(max_iter*steps):
                if k % steps == 0:
                    dE = self.epoch(lr_init, reg_cost, tol)
                    if self._scale_lr < 1e-4:
                        self.log.info("Learning rate too low, stop")
                        break

                out.write("\rTraining {}: {:7.2%} - {:10.3e}"
                          .format(self.name, k/(max_iter*steps), dE))
                out.flush()
                feed_dict = self._get_feed(batch_provider)
                # it = self.global_step.eval()
                feed_dict[self.lr] = self._scale_lr*lr_init  # *np.log(np.e+it)
                cost, _ = self.session.run(
                    [self._cost, self._train], feed_dict=feed_dict)
                if cost > 2*training_cost:
                    self.log.debug("Explode !! {} -  {:.4e}"
                                   .format(k, cost/training_cost))
                    self._scale_lr *= .95
                    for lyr in self.param_layers:
                        for p in lyr:
                            acc = self._optimizer.get_slot(p, 'accumulator')
                            if acc:
                                acc.initializer.run(session=self.session)
                    # self.restore()
                    self.import_param(self.mParams)
                    training_cost = self.session.run(self._cost,
                                                     feed_dict=feed_dict)
                else:
                    training_cost = cost


            self.epoch(lr_init, reg_cost, tol)
            # self.restore()
            self.import_param(self.mParams)
            self.writer.flush()
            print("\rTraining {}: {:7}".format(self.name, "done"))

            # Save the variables to disk.
            if save_model:
                save_path = self.saver.save(
                    self.session,
                    "save_exp/{}-{}.ckpt".format(model_name, self.n_layers),
                    global_step=self.global_step)
                self.log.info("Model saved in file: %s" % save_path)
