# train logistic regression on mnist dataest using lista

import numpy as np
import theano.tensor as T
import theano as K
import theano
import gzip, cPickle
from random import sample, seed
import os, sys
os.chdir('/home/dikai/PycharmProjects/sparse_lstm')
print(os.getcwd())

from sparse_lstm import Sparse_LSTM_wo_O_Gate_v2
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras import regularizers
from keras.callbacks import Callback
from keras.engine import Layer
from keras.optimizers import Adadelta

import matplotlib.pyplot as plt
from osdfutils import crino



output_dir = __file__[:-3]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
sys.setrecursionlimit(10000)

# load and normalize data
with gzip.open("data/mnist.pkl.gz",'rb') as f:
    train_set_mnist, valid_set_mnist, test_set_mnist = cPickle.load(f)
train_set_mnist_img, train_set_mnist_label = train_set_mnist
test_set_mnist_img, test_set_mnist_label = test_set_mnist
print('Original train set of mnist: ' + str(train_set_mnist_img.shape))
print('Original test set of mnist: ' + str(test_set_mnist_img.shape))
train_set_mnist_mean = train_set_mnist_img.mean(axis=0)
train_set_mnist_std = train_set_mnist_img.std(axis=0)
train_set_mnist_img -= train_set_mnist_mean
train_set_mnist_img /= train_set_mnist_std + 1e-10
test_set_mnist_img -= train_set_mnist_mean
test_set_mnist_img /= train_set_mnist_std + 1e-10

# shuffle data
total_img = np.vstack([train_set_mnist_img, test_set_mnist_img])
total_label = np.hstack([train_set_mnist_label, test_set_mnist_label])
np.random.seed(10023)
np.random.shuffle(total_img)
np.random.seed(10023)
np.random.shuffle(total_label)
n = total_img.shape[0]
train_set_mnist_img = total_img[:n//2]
train_set_mnist_label = total_label[:n//2]
test_set_mnist_img = total_img[n//2:]
test_set_mnist_label = total_label[n//2:]


# total # of epochs for training lista sparse encoder
epochs = 100
# batch size
btsz = 512
# learning rate
lr = 0.9
# momentum
momentum = 0.9
# learning rate decay
decay = 0.95
# number of batches per epoch
batches = train_set_mnist_img.shape[0]/btsz
# size of sparse vectors
sparse_shape = 14*14
# sparsity weight
lmbd = 0.1
L = 1.0
# number of iterations
layers = 10


def batch_generator_lista(images, batch_size, labels = None, yield_label=False):
    while True:
        s = np.array(sample(xrange(images.shape[0]), batch_size), dtype=np.int32)
        if yield_label:
            yield (images[s].copy(), labels[s].copy())
        else:
            yield images[s].copy()


print("LISTA -- Learned ISTA without ISTA")
print("Epochs", epochs)
print("Batches per epoch", batches)
Dinit_lista = {"shape": (sparse_shape, 28*28), "variant": "normal", "std": 0.1}
config_lista = {"D": Dinit_lista, "layers": layers, "L": L, "lambda": lmbd}
# normalize weights according to this config
norm_dic_lista = {"D": {"axis":1, "c": 1.}}
# threshold theta should be at least some value
thresh_dic_lista = {"theta": {"thresh": 1e-2}}
x_lista, params_lista, cost_lista, rec_lista, z_lista = crino.lista(config=config_lista, shrinkage=crino.sh)
grads_lista = T.grad(cost_lista, params_lista)
# training ...
settings_lista = {"lr": lr, "momentum": momentum, "decay": decay}
# ... with stochastic gradient + momentum
#updates = crino.momntm(params, grads, settings)#, **norm_dic)
updates_lista = crino.adadelta(params_lista, grads_lista, settings_lista)#, **norm_dic)
# ... normalize weights
updates_lista = crino.norm_updt(params_lista, updates_lista, todo=norm_dic_lista)
# ... make sure threshold is big enough
updates_lista = crino.max_updt(params_lista, updates_lista, todo=thresh_dic_lista)

train_lista = theano.function([x_lista], cost_lista, updates=updates_lista,
                            allow_input_downcast=True)
print 'done.'


# Rerun this cell if another full number of epochs should be trained.
generator_lista = batch_generator_lista(train_set_mnist_img, btsz)
hist = []
for epoch in xrange(epochs):
    cost = 0
    sz = 0
    for i in xrange(batches):
        cost += btsz*train_lista(generator_lista.next())
        sz += btsz
    hist.append([epoch, cost/sz])
hist = np.array(hist)
plt.plot(hist[:,0], hist[:,1])
plt.show()

# function to get sparse coding based on trained lista
sparse_lista = theano.function([x_lista], z_lista, allow_input_downcast=True)
# function to get reconstructed image from original image
reconstruct_lista = theano.function([x_lista], rec_lista, allow_input_downcast=True)
# function to get reconstructed image from sparse code
reconstruct_lista_2 = theano.function([z_lista], T.dot(z_lista, L*params_lista[0]), allow_input_downcast=True)






# batch generator to get (sparse_vector, label) pairs
def batch_generator(encoder=None, batch_size=512, img=None, label=None, n_classes=10):
    img = img.astype(np.float32)
    n_total = img.shape[0]
    while True:
        index = np.array(sample(xrange(n_total), batch_size), dtype=np.int)
        img_batch = img[index]
        x = encoder(img_batch)
        y = np.zeros((batch_size, n_classes), dtype=np.float32)
        y[np.arange(batch_size), label[index]] = 1.0
        yield x, y


# logistic classifier
def get_logistic_classifier(input_dim, output_dim, C=0.01):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax', W_regularizer=regularizers.l2(C)))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

n_epoch = 50
n_classes = 10

encoder = sparse_lista
bg = batch_generator(encoder, img=train_set_mnist_img, label=train_set_mnist_label, n_classes=n_classes)
# get sparse vector for test set
train_set_mnist_sparse_vec = encoder(train_set_mnist_img)
test_set_mnist_sparse_vec = encoder(test_set_mnist_img)

def get_1hot_lab(label, n_classes):
    y = np.zeros((label.shape[0], n_classes), dtype=np.float32)
    y[np.arange(label.shape[0]), label] = 1.0
    return y

train_set_mnist_label_1hot = get_1hot_lab(train_set_mnist_label, 10)
test_set_mnist_label_1hot = get_1hot_lab(test_set_mnist_label, 10)


class Eval_Callback(Callback):
    def __init__(self):
        super(Eval_Callback, self).__init__()
        self.acc_history_train = []
        self.acc_history_test = []

    def on_epoch_end(self, epoch, logs={}):
        print('on epoch %s end' % epoch)
        out_train = self.model.evaluate(train_set_mnist_sparse_vec, train_set_mnist_label_1hot, verbose=0)
        out_test = self.model.evaluate(test_set_mnist_sparse_vec, test_set_mnist_label_1hot, verbose=0)
        print('Train => %f, test => %f' % (out_train[1], out_test[1]))
        self.acc_history_train.append(out_train)
        self.acc_history_test.append(out_test)


callback = Eval_Callback()
classifier = get_logistic_classifier(input_dim=14*14, output_dim=10)
# history = classifier.fit_generator(bg, samples_per_epoch=len(train_set_mnist_img), nb_epoch=n_epoch, verbose=1, callbacks=[callback])
history = classifier.fit(train_set_mnist_sparse_vec, train_set_mnist_label_1hot, batch_size=512, nb_epoch=n_epoch, verbose=1, callbacks=[callback])
history = history.history

# plot results
plt.figure()
plt.plot([x[1] for x in callback.acc_history_train])
plt.plot([x[1] for x in callback.acc_history_test])
plt.legend(['train', 'test'])
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.savefig('lista_logistic_reg_mnist.png')
plt.show()