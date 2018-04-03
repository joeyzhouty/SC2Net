import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from deepLearningModel.resnet50 import ResNet50
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator

def loadmnist(N):
    import gzip,cPickle
    with gzip.open("/data/home/dujw/darse/data_handlers/mnist.pkl.gz",'rb') as f:
        train_set_mnist, valid_set_mnist, test_set_mnist = cPickle.load(f)
    train_set_mnist_img, train_set_mnist_label = train_set_mnist
    return train_set_mnist_img[:N],train_set_mnist_label[:N]
def n2(x):
    import numpy as np
    return np.sum(x*x)


def soft_thresholding(x, theta):
    '''Return the soft-thresholding of x with theta
    '''
    import numpy as np
    return np.sign(x)*np.maximum(0, np.abs(x) - theta)


def drow_mnist(origi,input):
    input = input.reshape(input.shape[0],28,28)
    input = scope_255(input)
    origi = origi.reshape(origi.shape[0],28,28)
    origi = scope_255(origi)
    input = input - origi
    from matplotlib import pyplot as plt
    plt.figure(figsize=(15,8))
    plt.subplot(121)
    plt.imshow(combineArray(input),cmap='gray')
    plt.subplot(122)
    plt.imshow(combineArray(origi),cmap='gray')
    # plt.show()



def drow_cifar(origi,input):
    import cv2
    input = input.reshape(input.shape[0],3,32,32)
    input = input.swapaxes(3,1)
    # input = scope_255(input)
    origi = origi.reshape(origi.shape[0],3,32,32)
    origi = origi.swapaxes(3,1)
    # origi = scope_255(origi)
    # input = np.abs(input - origi)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(15,8))
    plt.subplot(121)
    input = combineArray(input)
    input_cp = np.copy(input)
    # input[:,:,0] = input_cp[:,:,2]
    # input[:,:,2] = input_cp[:,:,0]
    input = np.rot90(input,3,(0,1))
    input = input.astype(np.uint8)
    plt.imshow(input)
    plt.subplot(122)
    origi = combineArray(origi)
    origi_cp = np.copy(origi)
    origi[:,:,0] = origi_cp[:,:,0]
    origi[:,:,1] = origi_cp[:,:,1]
    origi[:,:,2] = origi_cp[:,:,2]
    origi = np.rot90(origi,3,(0,1))
    origi = origi.astype(np.uint8)
    plt.imshow(origi)
    # plt.show()

def combineArray(inp):
    # return inp.reshape(280,28)0)
    return np.vstack((np.hstack((inp[j+i*10] for j in range(10))) for i in range(10)))
def scope_255(inp):
    # inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))*255
    inp = (inp-np.min(inp))*255
    inp = inp.astype(np.int16)
    return inp 
def start_handler(logger, log_lvl, out=None):
    """Add a StreamHandler to logger is no handler as been started. The default
    behavior send the log to stdout.

    Parameters
    ----------
    logger: Logger object
    log_lvl: int
        Minimal level for a log message to be printed. Refers to the levels
        defined in the logging module.
    out: writable object
        Should be an opened writable file object. The default behavior is to
        log tht messages to STDOUT.
    """
    import logging
    logger.setLevel(log_lvl)
    if len(logger.handlers) == 0:
        if out is None:
            from sys import stdout as out

        ch = logging.StreamHandler(out)
        ch.setLevel(log_lvl)
        formatter = logging.Formatter('\r[%(name)s] - %(levelname)s '
                                      '- %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
def creat_mnist_feature():
    dir_mnist = 'save_exp/mnist'
    mnist = input_data.read_data_sets(dir_mnist, one_hot=True)
    im,label = mnist.train.next_batch(50000)
    # model = ResNet50(weights='imagenet',include_top=False) 
    model = getModel("weights/mnist_weight.h5")
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()

    feat = feat_extract(model,im)
    import pdb
    pdb.set_trace()
    location = "save_exp/mnist/feature.pkl"
    save_feature(feat,label,location)

def save_feature(feat,label,location):
    dict = {"feature":feat,"label":label}
    output = open(location,"wb")
    import pickle
    pickle.dump(dict,output)
    output.close()
def feat_extract(model,img):
    newimg = []
    for each in img:
        # each = np.repeat(cv2.resize(each,(224,224)).reshape(224,224,1),3,axis=2)
        each = each.reshape((28,28,1))
        newimg.append(each)

    newimg = np.array(newimg)
    feat = model.predict(newimg)
    feat = np.reshape(feat,(feat.shape[0],-1))
    return feat
def getModel(load_model=None):

    model = Sequential()

    model.add(Convolution2D(8,(2,2),padding='same',input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Convolution2D(8,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(16,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Convolution2D(32,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(32,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(32,activation="relu"))  #256
    model.add(Dropout(0.5))
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation="softmax"))

    print model.summary()
    if load_model!=None:
        model.load_weights(load_model)
        print "model of {} loaded".format(load_model)

    return model



def trainModel():
    model = getModel()
    dir_mnist = 'save_exp/mnist'
    mnist = input_data.read_data_sets(dir_mnist, one_hot=True)
    im,label = mnist.train.next_batch(50000)
    im = im.reshape((im.shape[0],28,28,1))
    model.compile(loss="binary_crossentropy",optimizer="Adadelta",metrics=['accuracy'])
    model.fit(im[:45000],label[:45000],batch_size=32,nb_epoch=25,verbose=2,validation_data=(im[45000:],label[45000:]))
    model.save_weights("weights/mnist_weight.h5")
def mul_coher(matri):
    columns = matri.shape[1]
    result = np.copy(matri).astype(np.float64)
    for col in range(columns):
        result[:,col] = result[:,col]/np.sqrt(np.matmul(np.transpose(result[:,col]),result[:,col]))
    col = np.matmul(np.transpose(result),result)
    for i in range(np.min(np.array(col.shape))):
        col[i,i]=0
    return np.max(col)
if __name__ == '__main__':
    for i in range(1000):
        print mul_coher(np.random.normal(size=[100,20]))
