import tensorflow as tf
import numpy as np

import utils

import random
import argparse
import sys
import keras.backend as K 
from sys import stdout as out

class SLSTM():

    def __init__(self, state_size, num_classes, num_layers,
            ckpt_path='../weight/',
            model_name='slstm'):

        self.input_size = 2048
        self.state_size = state_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.ckpt_path = ckpt_path
        self.model_name = model_name

        # build graph ops
        def __graph__():
            tf.reset_default_graph()
            # inputs
            xs_ = tf.placeholder(shape=[None,None,self.input_size], dtype=tf.float32)
            # ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
            #
            # embeddings
            # embs = tf.get_variable('emb', [num_classes, state_size])
            # rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            rnn_inputs = xs_ 
            #
            # initial hidden state
            init_state = tf.placeholder(shape=[2, self.num_layers, None,state_size],
                    dtype=tf.float32, name='initial_state')
            # initializer
            xav_init = tf.contrib.layers.xavier_initializer
            # params
            W = tf.get_variable('W', 
                    shape=[4, self.state_size,self.state_size], initializer=xav_init())
            U = tf.get_variable('U', 
                    shape=[4, self.input_size,self.state_size], initializer=xav_init())
            b = tf.get_variable('b', shape=[3, self.state_size], initializer=tf.constant_initializer(0.))
            D = tf.get_variable('D', 
                    shape=[self.state_size,self.input_size], initializer=xav_init())
            ####
            # step - LSTM
            def step(prev, x):
                # gather previous internal state and output state
                st_1, ct_1 = tf.unstack(prev)

                # iterate through layers
                st, ct = [], []
                inp = x
                for i in range(num_layers):
                    ####
                    # GATES
                    #
                    #  input gate
                    ig = tf.sigmoid(tf.matmul(inp, U[0]) + tf.matmul(st_1[i],W[0])+b[0])
                    # ig = tf.sigmoid(tf.matmul(inp, U[0]))

                    #  forget gate
                    fg = tf.sigmoid(tf.matmul(inp, U[1]) + tf.matmul(st_1[i],W[1])+b[1])
                    #  output gate
                    # og = K.epsilon() * tf.sigmoid(tf.matmul(inp, U[2]) + tf.matmul(st_1[i],W[2])) + 1.0
                    #  gate weights
                    # g = tf.tanh(tf.matmul(inp, U[3]) + tf.matmul(st_1[i],W[3]))

                    g = tf.matmul(inp, U[3]) + tf.matmul(st_1[i],W[3])
                    ###
                    # new internal cell state
                    ct_i = fg * ct_1[i] + ig * g 
                    # output state
                    # st_i = (tf.tanh(ct_i-b[2])+tf.tanh(ct_i+b[])) 
                    st_i = tf.nn.relu(ct_i-b[2]) - tf.nn.relu(-ct_i-b[2]) 
                    inp = st_i
                    st.append(st_i)
                    ct.append(ct_i)
                return tf.stack([st, ct])


            ###
            states = tf.scan(step, 
                    tf.transpose(rnn_inputs,[1,0,2]),
                    initializer=init_state)
            #

            ####
            # get last state before reshape/transpose
            last_state = states[:,0,-1,:,:]

            ct_state = states[-1,1,:,:,:]
            # ct_state = tf.reshape(ct_state,[self.num_layers,-1,self.state_size])

            st_state = states[-1,0,:,:,:]
            # st_state = tf.reshape(st_state,[self.num_layers,-1,self.state_size])


            last_state = tf.reshape(last_state,[-1,self.state_size])
            ####
            # transpose/slice -> pick st from [ct, st] -> pick st[-1] from st
            # flatten states to 2d matrix for matmult with V
            #states_reshaped = tf.reshape(states, [st_shp[0] * st_shp[1], st_shp[2]])
            # optimization
            # X = tf.reshape(xs_ ,[-1,self.state_size])
            X = tf.reshape(tf.transpose(rnn_inputs,[1,0,2]),[-1,self.input_size])
            # rec = tf.matmul(last_state,tf.transpose(U[3]))
            rec = tf.matmul(last_state,D)
            
            Zk = last_state

            with tf.name_scope("norm_2"):
                Er = tf.multiply(
                    tf.constant(.5, dtype=tf.float32),
                    tf.reduce_mean(tf.reduce_sum(tf.squared_difference(rec, X),
                                                 reduction_indices=[1])))

            with tf.name_scope("norm_1"):
                l1 = 0.2*tf.reduce_mean(tf.reduce_sum(
                    tf.abs(Zk), reduction_indices=[1]))

            loss = tf.add(Er,l1)
            # loss = Er
            # train_op = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(loss)
            train_op = tf.train.AdadeltaOptimizer().minimize(loss)
            #
            # expose symbols
            self.W = U[3]
            self.xs_ = xs_
            self.loss = loss
            self.train_op = train_op
            self.last_state = last_state
            self.init_state = init_state
            self.ct_state = ct_state
            self.st_state = st_state
        ##### 
        # build graph
        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    ####
    # training
    def train(self, train_set, epochs=EPOCHS):
        # training session
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            if CONTINUE:
            # restore session
                ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
                saver = tf.train.Saver()
                if ckpt and ckpt.model_checkpoint_path:
                    print "model load ",ckpt.model_checkpoint_path
                    saver.restore(sess, ckpt.model_checkpoint_path)


            train_loss = 0
            # batch_size = 210
            batch_size = 21*BATCH_SIZE
            steps = 15328/(BATCH_SIZE*TIME_STEP)+1
            st = np.zeros([1, self.num_layers, batch_size, self.state_size])
            ct = np.zeros([1, self.num_layers, batch_size, self.state_size])
            try:
                for i in range(epochs):
                    for j in range(steps):
                        xs,ys = train_set.next()
                        xs = np.reshape(xs,[TIME_STEP,-1,xs.shape[-1]])
                        xs = np.swapaxes(xs,0,1)
                        #batch,timestep.features
                        batch_size = xs.shape[0]
                        _,ct,st, train_loss_ = sess.run([self.train_op,self.ct_state,self.st_state, self.loss], feed_dict = {
                                self.xs_ : xs,
                                self.init_state: np.concatenate((st,ct))
                            })
                        #ct,st need to be reshaped 
                        ct = ct.reshape([1, self.num_layers, batch_size, self.state_size])
                        st = st.reshape([1, self.num_layers, batch_size, self.state_size])
 
                        train_loss += train_loss_
                        print train_loss_
                        # out.write("current loss is {}".format(train_loss_))
                        # out.flush()
                    print('[{}] loss : {}'.format(i,train_loss/steps))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
            #
            # training ends here; 
            #  save checkpoint
            states = np.concatenate((st,ct))
            np.save("../weight/state.npy",states)
            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
    ####
    # generate characters
    def generate(self):
        #
        # generate text
        #
       # start session
        batch_size = 1
        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())
            #
            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                print "model load ",ckpt.model_checkpoint_path
                saver.restore(sess, ckpt.model_checkpoint_path)



            xs,ys = get_data.get_testing()
            xs = xs
            st = np.zeros([1, self.num_layers, batch_size, self.state_size])
            ct = np.zeros([1, self.num_layers, batch_size, self.state_size])

            st,ct = np.load("../weight/state.npy")[:,:,-1,:]
            ct = ct.reshape([1, self.num_layers, batch_size, self.state_size])
            st = st.reshape([1, self.num_layers, batch_size, self.state_size])
 
            ls_total = np.zeros(xs.shape[:2])
            for i in range(21):
                st,ct = np.load("../weight/state.npy")[:,:,-1,:]
                ct = ct.reshape([1, self.num_layers, batch_size, self.state_size])
                st = st.reshape([1, self.num_layers, batch_size, self.state_size])
                st = np.zeros([1, self.num_layers, batch_size, self.state_size])
                ct = np.zeros([1, self.num_layers, batch_size, self.state_size])


 
                for index,each in enumerate(xs):
                    # if index%10 ==0:
                        # st = np.zeros([1, self.num_layers, batch_size, self.state_size])
                        # ct = np.zeros([1, self.num_layers, batch_size, self.state_size])


                    xs_ = each[i]
                    xs_ = xs_.reshape(1,1,xs_.shape[-1])
                    feed_dict = {self.xs_:xs_,
                            # self.init_state : np.zeros([2, self.num_layers, batch_size, self.state_size])
                            self.init_state: np.concatenate((st,ct))
                            }

                    st,ct, loss = sess.run([self.st_state,self.ct_state, self.loss], feed_dict=feed_dict)
                    if loss > 1000000:
                        import pdb
                        pdb.set_trace()
                    ct = ct.reshape([1, self.num_layers, batch_size, self.state_size])
                    st = st.reshape([1, self.num_layers, batch_size, self.state_size])
                    
                    print "st mean is ",np.mean(st)
                    print "loss is ",loss
                    ls_total[index,i] = loss
            np.save("../weight/loss_1380_no_timestep_lay6.npy",ls_total)
            # generate operation
            '''
            words = [current_word]
            state = None
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : state_}
                else:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : np.zeros([2, self.num_layers, 1, self.state_size])}
                #
                # forward propagation
                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
                # 
                # set flag to true
                state = True
                # 
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)
        ########
        '''
        # return the list of words as string
        # return separator.join([idx2w[w] for w in words])

### 
# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Stacked Long Short Term Memory RNN for Text Hallucination, built with tf.scan')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store_true',
                        help='generate text')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate')
    args = vars(parser.parse_args())
    return args


###
# main function
if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    #
    # fetch data
    # X, Y, idx2w, w2idx= data.load_data('data/paulg/')
    X = get_data.get_training()
    Y = np.ones_like(X)
    X = np.array(X)
    seqlen = X.shape[0]

    # create the model

    model = SLSTM(state_size = STATE_SIZE, model_name="lstm_ped2",num_classes=5, num_layers=LAYERS)
    if args['train']:
        # get train set
        train_set = utils.rand_batch_gen(X, Y ,batch_size=BATCH_SIZE*TIME_STEP)
        #
        # start training
        model.train(train_set)
    elif args['generate']:
        # call generate method
        text = model.generate()
        #########
        # text generation complete
        #
        print('______Generated Text_______')
        print(text)
        print('___________________________')
