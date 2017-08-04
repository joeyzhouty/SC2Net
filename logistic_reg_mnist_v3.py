# train logistic regression on mnist dataest

import numpy as np
import theano.tensor as T
import theano as K
import gzip, cPickle
import matplotlib.pyplot as plt
from random import sample, seed
import os, sys
os.chdir('data/sparse_lstm')
print(os.getcwd())

from sparse_lstm import Sparse_LSTM_wo_O_Gate_v2
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras import regularizers
from keras.callbacks import Callback
from keras.engine import Layer
from keras.optimizers import Adadelta

import matplotlib.pyplot as plt


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



# logistic classifier
def get_logistic_classifier(input_dim, output_dim, C=0.01):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax', W_regularizer=regularizers.l2(C)))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# batch generator
def batch_generator(batch_size=512, img=None, label=None, n_classes=10):
    img = img.astype(np.float32)
    n_total = img.shape[0]
    while True:
        index = np.array(sample(xrange(n_total), batch_size), dtype=np.int)
        img_batch = img[index]
        y = np.zeros((batch_size, n_classes), dtype=np.float32)
        y[np.arange(batch_size), label[index]] = 1.0
        yield img_batch.copy(), y

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
        out_train = self.model.evaluate(train_set_mnist_img, train_set_mnist_label_1hot, verbose=0)
        out_test = self.model.evaluate(test_set_mnist_img, test_set_mnist_label_1hot, verbose=0)
        print('Train => %f, test => %f' % (out_train[1], out_test[1]))
        self.acc_history_train.append(out_train)
        self.acc_history_test.append(out_test)

callback = Eval_Callback()


model = get_logistic_classifier(28*28, 10)
history = model.fit(train_set_mnist_img, train_set_mnist_label_1hot, batch_size=256, nb_epoch=30, verbose=1, callbacks=[callback])

plt.figure()
plt.plot([x[1] for x in callback.acc_history_train])
plt.plot([x[1] for x in callback.acc_history_test])
plt.legend(['train', 'test'])
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.savefig('logistic_reg_mnist.png')
plt.show()