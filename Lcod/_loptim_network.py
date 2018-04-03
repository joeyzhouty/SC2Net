from __future__ import division
import time 
import numpy as np
import os.path as osp
import os
import tensorflow as tf
from sys import stdout as out
from sklearn.linear_model import LogisticRegression as lgr
from sklearn.metrics import accuracy_score
import sys 
sys.path.append("/data/home/dujw/darse/data_handlers/recon_sparse")
from evaluate import cal_acry


TMP_DIR = osp.join('~/tmp', 'TensorBoard')
if not osp.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

class _LOptimNetwork(object):
    """Base class for adaptive learning networks"""
    def __init__(self, n_layers, name='Loptim', shared=False,supervised = False, warm_param=[],
                 gpu_usage=1, reg_scale=1, init_value_ada=1e-2, exp_dir=None):
        self.n_layers = n_layers
        self.shared = shared
        self.warm_param = warm_param
        self.gpu_usage = gpu_usage
        self.reg_scale = reg_scale
        self.init_value_ada = init_value_ada
        self.exp_dir = exp_dir if exp_dir else 'default'
        self.name = name
        self.DD = 0
        self.CC = 0
        self.supervised =supervised
        self._construct()
        self.reset()

    def test_accuracy(self,pb,lmbd=0.1):

        test_x,test_y = pb.get_batch_with_label(8000)
        test_y = np.argmax(test_y,axis=1)
        zs_test = np.zeros((8000,self.D.shape[0]))
        feed_final = {"Z": zs_test[:6000], "X": test_x[:6000], "lmbd": lmbd}
        train_data = self.output(**feed_final)

        lgrs = lgr()
        lgrs.fit(train_data,test_y[:6000])

        # test_x,test_y = pb.get_batch_with_label(1000)
        # test_y = np.argmax(test_y,axis=1)
        feed_final = {"Z": zs_test[:2000], "X": test_x[6000:], "lmbd": lmbd}
        lis_out = self.output(**feed_final)
        y_pre = lgrs.predict(lis_out)
        
        return accuracy_score(test_y[6000:],y_pre)

    def softmax(self,x):
            scoreMatExp = np.exp(np.asarray(x))
            return scoreMatExp / scoreMatExp.sum(0)
    def test_accuracy_self(self,pb,lmbd=0.1):

        test_x,test_y = pb.get_test_with_label(2000)
        zs_test = np.zeros((2000,self.D.shape[0]))

        feed_final = {"Z": zs_test, "X": test_x,"label":test_y,"lmbd": lmbd}
        lis_out = self.output(**feed_final)
        with self.session.as_default():
            CC = self.CC.eval()
            y_pre = []
            for i in lis_out:
                y_pre.append(self.softmax(np.matmul(i,CC)))

        y_pre = np.array(y_pre) 

        test_y = np.argmax(test_y,axis=1)
        y_pre = np.argmax(y_pre,axis=1)
        return accuracy_score(test_y,y_pre)


    def test_loss(self,pb,lmbd=0.1):

        test_x,test_y = pb.get_batch_with_label(8000)
        zs_test = np.zeros((8000,self.D.shape[0]))
        feed_final = {"Z": zs_test, "X": test_x, "lmbd": lmbd}
        train_data = self.output(**feed_final)
        from numpy.linalg import norm 
        sum = 0.0
        # for i in range(8000):
           # sum+=norm(train_data[i]-test_y[i],2)
        # return sum/8000
        for i in range(8000):
            sum += cal_acry(train_data[i],test_y[i])
        return sum/8000

    def reconstruct(self,pb,lmbd=0.1):
        # test_x  = pb.get_batch(100,shuffle=False)[0]
        from utils import loadmnist 
        test_x = loadmnist(100)[0]
        # test_x = pb.get_truth(100)

        zs_test = np.zeros((100,self.D.shape[0]))
        feed_final = {"Z": zs_test, "X": test_x, "lmbd": lmbd}
        train_data = self.output(**feed_final)
        recu_x = []
        with self.session.as_default():
            DD = self.DD.eval()
            recu_x = []
            for i in train_data:
                rec = np.matmul(i,DD)
                recu_x.append(rec)
            recu_x = np.array(recu_x)
        from utils import drow_mnist
        from utils import drow_cifar
        if test_x.shape[1]>1000:
            drow_cifar(test_x,recu_x)
        else:
            drow_mnist(test_x,recu_x)
        return recu_x

    def test_mul(self,pb,lmbd=0.1):
        with self.session.as_default():
            DD = self.DD.eval()
        from utils import mul_coher
        return mul_coher(DD)







    def _construct(self):
        """Construct the network by calling successively the layer method
        """
        K, p = self.D.shape
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Declare the training variables
            self.lr = tf.placeholder(dtype=tf.float32,
                                     name='learning_rate')
            self.global_step = tf.contrib.framework.create_global_step()
            # Make sure that regularization can be fetched
            tf.add_to_collection("regularisation",
                                 tf.constant(0., dtype=tf.float32))

            # Construct the first layer from the inputs of the network
            inputs = self._get_inputs()

            # construct classification matrix C
            if self.supervised == True:
                with tf.name_scope("supervised"):
                    Wc = tf.Variable(tf.constant(np.ones([K,10]).astype(np.float32),shape=[K,10]),name="Wc")



            with tf.name_scope("layer_0"):
                outputs, params = self._layer(inputs, id_layer=0)
                tf.add_to_collection("layer_costs", self._get_cost(outputs))
            self.param_layers = [params]
            if not self.shared:
                params = None
            # Construct the next layers by chaining the outputs
            for k in range(1, self.n_layers):
                with tf.name_scope("layer_{}".format(k)):
                    outputs, params = self._layer(outputs, params=params,
                                                  id_layer=k)
                    tf.add_to_collection("layer_costs",
                                         self._get_cost(outputs))

                if not self.shared:
                    self.param_layers += [params]
                    params = None

            # Construct and store the output/cost operation for the network
            self._output = self._get_output(outputs)
            with tf.name_scope("Cost"):
                self._cost = self._get_cost(outputs)

            c_val = tf.Variable(tf.constant(0, dtype=tf.float32),
                                name='c_val')
            tf.summary.scalar('cost_val', self._cost-c_val)
            tf.summary.scalar('learning_rate', self.lr)
            self.feed_map['c_val'] = c_val

            # Construct the training step.
            self._train = self._mk_training_step()

            self.var_init = tf.initialize_all_variables()
            self.saver = tf.train.Saver(
                var_list=[pl for pp in self.param_layers
                          for pl in pp if pl is not None] + [self.global_step],
                max_to_keep=1)

            self.summary = tf.summary.merge_all()
            self.logdir = self._mk_logdir()
            self.writer = tf.summary.FileWriter(self.logdir, self.graph,
                                                 flush_secs=30)

    def _mk_logdir(self):
        logdir = osp.join(TMP_DIR, self.exp_dir, self.name)
        if osp.exists(logdir):
            import shutil
            shutil.rmtree(logdir)
        else:
            if not osp.exists(TMP_DIR):
                import os
                os.mkdir(TMP_DIR)
            dir_name = osp.join(TMP_DIR, self.exp_dir)
            if not osp.exists(dir_name):
                import os
                os.mkdir(dir_name)
        return logdir

    def _layer(self, input, params=None, id_layer=0):
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
        raise NotImplementedError("{} must implement the _layer method"
                                  "".format(self.__class__))

    def _get_inputs(self):
        """Construct the placeholders used for the network inputs, to be passed
        as entries for the first layer.

        Return
        ------
        outputs: tuple of tensors (n_in) passed as entries to construct the 1st
                 layer of the network.
        """
        raise NotImplementedError("{} must implement the _get_inputs method"
                                  "".format(self.__class__))

    def _get_output(self, outputs):
        """Select the output of the network from the outputs of the last layer.
        This permits to select the result from the self.output methods.
        """
        return self.graph.get_tensor_by_name(
            "layer_{}/output:0".format(self.n_layers-1))

    def _get_feed(self, batch_provider):
        """Construct the feed dictionary from the batch provider

        This method will be use to feed the network at each step of the
        optimization from the batch provider. It will put in correspondance
        the tuple return by the batch_provider and the input placeholders.
        """
        raise NotImplementedError("{} must implement the _get_feed method"
                                  "".format(self.__class__))

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
        """
        raise NotImplementedError("{} must implement the _get_cost method"
                                  "".format(self.__class__))

    def _mk_training_step(self):
        """Function to construct the training steps and procedure.

        This function returns an operation to iterate and train the network.
        By default, an AdagradOptimizer is used.
        """
        # Training methods
        _reg = tf.add_n(tf.get_collection("regularisation"))
        # self._optimizer = tf.train.AdagradOptimizer(
            # self.lr, initial_accumulator_value=self.init_value_ada)

        self._optimizer = tf.train.AdadeltaOptimizer(1)
        grads = self._optimizer.compute_gradients(
            self._cost + self.reg_scale*_reg)
        # for grad, var in grads:
        #     if grad is not None:
        #         tf.histogram_summary(var.op.name + '/gradients', grad)
        return self._optimizer.apply_gradients(grads,
                                               global_step=self.global_step)

    def reset(self):
        """Reset the state of the network."""
        if hasattr(self, 'session'):
            self.session.close()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_usage
        self.session = tf.Session(graph=self.graph, config=config)
        self.cost_val = []
        self._scale_lr = 1
        self.mE = 1e100
        self.session.run(self.var_init)

    def terminate(self):
        self.session.close()

    def restore(self):
        ckpt = tf.train.latest_checkpoint(osp.dirname(self.logdir))
        self.saver.restore(self.session, ckpt)

    def save(self, savefile=None):
        if savefile is None:
            savefile = "{}.ckpt".format(self.logdir)
        save_path = self.saver.save(self.session, savefile,
                                    global_step=self.global_step)
        self.log.info("Model saved in file: %s" % save_path)

    def train(self, batch_provider, feed_val, max_iter, steps, lr_init=.01,
              tol=1e-5, reg_cost=15, model_name='loptim', save_model=False):
        """Train the network
        """
        self._feed_val = self._convert_feed(feed_val)
        self._last_downscale = -reg_cost

        # use to count the time
        timecount = 0

        with self.session.as_default():
            training_cost = self._cost.eval(feed_dict=self._feed_val)
            timecount = time.time()
            for k in range(max_iter*steps):
                if k % steps == 0:
                    print "used time is ",round(time.time()-timecount,2)
                    timecount = time.time()
                    dE = self.epoch(lr_init, reg_cost, tol)
                    if self._scale_lr < 1e-4:
                        self.log.info("Learning rate too low, stop")
                        break

                out.write("\rTraining {}: {:7.2%} - {:10.3e}"
                          .format(self.name, (k+0.0)/(max_iter*steps), dE))
                out.flush()
                feed_dict = self._get_feed(batch_provider)
                # it = self.global_step.eval()
                feed_dict[self.lr] = self._scale_lr*lr_init  # *np.log(np.e+it)
                cost, _ = self.session.run(
                    [self._cost, self._train], feed_dict=feed_dict)

                if cost > 2*training_cost:
                    self.log.info("Explode !! {} -  {:.4e}"
                                  .format(k, cost/training_cost))
                    self._scale_lr *= .9
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

    def epoch(self, lr_init, reg_cost, tol):
        it = self.global_step.eval()
        self._feed_val[self.lr] = self._scale_lr*lr_init  # *np.log(np.e+it)
        cost, summary = self.session.run(
            [self._cost, self.summary], feed_dict=self._feed_val)
        print "cost is ",cost
        self.cost_val += [cost]
        self.writer.add_summary(summary, global_step=it)

        # store the best model on validation set
        # it is used to reload the model when the optim fails
        if self.mE > cost:
            # self.save()
            self.mParams = self.export_param()
            self.mE = cost

        dE = 1
        if len(self.cost_val) > 2*reg_cost:
            dE = (1 - np.mean(self.cost_val[-reg_cost:]) /
                  np.mean(self.cost_val[-2*reg_cost:-reg_cost]))
            ds = self._last_downscale
            if dE < tol and (it - ds) >= (reg_cost // 2):
                self.log.debug("Downscale lr at iteration {: 4} -"
                               " ({:10.3e})".format(it, dE))
                self._scale_lr *= .95
                self._last_downscale = it
        return cost - self._feed_val[self.feed_map['c_val']]

    def output(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        with self.session.as_default():
            return self._output.eval(feed_dict=feed_dict)

    def cost(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        return self.session.run(self._cost, feed_dict=feed_dict)

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            if k in self.feed_map.keys():
                _feed[self.feed_map[k]] = v
        return _feed

    def export_param(self, n_layer=None):
        export = []
        with self.session.as_default():
            for params in self.param_layers[:n_layer]:
                pl = []
                for p in params:
                    pl += [p.eval() if p is not None else None]
                export += [pl]
        return export

    def import_param(self, wp, n_layer=None):
        to_run = []
        with self.session.as_default():
            with self.graph.as_default():
                for wpl, params in zip(wp, self.param_layers[:n_layer]):
                    for w, p in zip(wpl, params):
                        if p is not None:
                            to_run += [p.assign(tf.constant(w))]

            self.session.run(to_run)
