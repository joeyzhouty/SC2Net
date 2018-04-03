import numpy as np
import gzip
import cPickle

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
train_set_mnist_img /= train_set_mnist_std + 1e-7
test_set_mnist_img -= train_set_mnist_mean
test_set_mnist_img /= train_set_mnist_std + 1e-7

seed = 10023
# shuffle data
total_img = np.vstack([train_set_mnist_img, test_set_mnist_img])
total_label = np.hstack([train_set_mnist_label, test_set_mnist_label])
np.random.seed(seed)
np.random.shuffle(total_img)
np.random.seed(seed)
np.random.shuffle(total_label)
n = total_img.shape[0]
train_set_mnist_img = total_img
train_set_mnist_label = total_label
# test_set_mnist_img = total_img[n//2:]
# test_set_mnist_label = total_label[n//2:]