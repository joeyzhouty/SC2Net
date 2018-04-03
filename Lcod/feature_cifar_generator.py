#python scripts
__author__='Du Jiawei'
#Descrption:
import numpy as np
from sklearn.utils import shuffle as shuf
from keras.datasets import cifar10
import pickle
class CifarFeatureGenerator():
    def __init__(self, D, lmbd, batch_size=100, dir_mnist='save_exp_cifar',seed=0):
        self.D = np.array(D)
        self.K, self.p = D.shape
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.patch_size = int(np.sqrt(self.p))
        self.dict = unpickle("/data/home/dujw/darse/save_exp/cifar/cifar_feature.pkl")
        self.im_t = self.dict['feature_t']
        self.label_t = self.dict['label_t']
        self.seed = seed
        self.im =  self.dict['feature']
        self.label = self.dict['label']
    def get_batch(self, N=None,shuffle=True):
        if N is None:
            N = self.batch_size
        # im = self.im
        # batch = shuf(im,random_state=np.random.randint(1,10000))
        # X = batch[:N].reshape(N, -1)
        X,label = self.get_batch_with_label(N,shuffle)
        z = np.zeros((N, self.K))
        return X, z, z, label,self.lmbd
    def get_truth(self,N):
        import pdb
        pdb.set_trace()
        return self.dict['data'][:N]
    def get_batch_with_label(self,N=None,shuffle=True):
        from keras.utils import np_utils
        if N is None:
            N = self.batch_size
        im = self.im
        if shuffle ==True:
            batch,label = shuf(im,self.label,random_state=np.random.randint(1,10000))
        else:
            batch,label = im,self.label
        X = batch[:N].reshape(N, -1)
        label =label[:N]
        label = np_utils.to_categorical(label,10)
        return X,label   
    def get_test_with_label(self,N=None,shuffle=True):
        from keras.utils import np_utils
        if N is None:
            N = self.batch_size
        im = self.im_t
        if shuffle ==True:
            batch,label = shuf(im,self.label_t,random_state=np.random.randint(1,10000))
        else:
            batch,label = im,self.label
        X = batch[:N].reshape(N, -1)
        label =label[:N]
        label = np_utils.to_categorical(label,10)
        return X,label
    def get_test(self,N=None):
        X, z, z, label,self.lmbd  = self.get_batch(N=N)
        return X, z, z,self.lmbd  



def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def getfeature():
    dict = unpickle("/data/home/dujw/darse/save_exp/cifar/data_batch_3")
    data = dict["data"]
    data = data.reshape(data.shape[0],3,32,32)
    data = data.swapaxes(3,1)
    data = np.rot90(data,3,(1,2))

    from matplotlib import pyplot as plt
    plt.figure(figsize=(15,8))
    # data = data.astype(np.float64)
    from deepLearningModel.imagenet_utils import preprocess_input
    # data =preprocess_input(data)
    # data =data /255
    model = getModel2() 
    model.compile(loss="categorical_crossentropy",optimizer="Adadelta",metrics=['accuracy'])
    label = dict["labels"]
    from keras.utils import np_utils, generic_utils
    label = np_utils.to_categorical(label,10)
    from  sklearn.model_selection import train_test_split

    # from mnist_problem_generator import MnistProblemGenerator
    # pb = MnistProblemGenerator(np.random.random([12,12]),0.1)
    # data,label = pb.get_batch_with_label(30000)
    # data = data.reshape(data.shape[0],28,28,1).astype(np.float32)

    x,x_t, y ,y_t = train_test_split(data, label, test_size=0.2, random_state=42)
    (x,y),(x_t,y_t) = cifar10.load_data()
    x = x.astype(np.float32)/255
    x_t = x_t.astype(np.float32)/255
    y = np_utils.to_categorical(y,10)
    y_t = np_utils.to_categorical(y_t,10)

    from keras.preprocessing.image import ImageDataGenerator
    generator = ImageDataGenerator(rotation_range=10,width_shift_range=0.03,height_shift_range=0.03,zoom_range=0.03)

    # model.fit(x,y,batch_size=32,nb_epoch=25,verbose=2,validation_data=(x_t,y_t))
    model.fit_generator(generator.flow(x,y),steps_per_epoch=400,epochs=200,validation_data=(x_t,y_t))
    # model.fit(x,y,batch_size=32,nb_epoch=200,verbose=2,validation_data=(x_t,y_t))
    import pdb
    pdb.set_trace()
    model.save_weights("../weights/fcn_weight.h5")

def createFeature():
    model =getModel2("../weights/fcn_weight.h5")
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.compile(loss="categorical_crossentropy",optimizer="Adadelta",metrics=['accuracy'])
    from keras.utils import np_utils, generic_utils
    from  sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression as lgr
    from sklearn.metrics import accuracy_score
    (x,y),(x_t,y_t) = cifar10.load_data()
    x = x.astype(np.float32)/255
    x_t = x_t.astype(np.float32)/255
    # y = np_utils.to_categorical(y,10)
    # y_t = np_utils.to_categorical(y_t,10)
    y_t = y_t.reshape(-1)
    y = y.reshape(-1)

    feat = model.predict(x[:10000])
    feat_t = model.predict(x_t)

    mydict = {"feature_t":feat_t,"label_t":y_t,"feature":feat,"label":y[:10000]}
    output = open('cifar_feature.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()




def getModel(weight=None):
    from keras import backend as K
    K.set_image_dim_ordering('tf')
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import SGD, RMSprop


    model = Sequential()

    model.add(Convolution2D(32,(2,2),padding='same',input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Convolution2D(32,(2,2),padding='same'))
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


    model.add(Convolution2D(64,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(64,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(128,(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(512,activation="relu"))  #256
    model.add(Dropout(0.5))
    # model.add(Dense(256,activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(10,activation="softmax"))
    # model.add(Activation("softmax"))
    # model.add(Convolution2D(1,(7,7)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))


    print model.summary()
    if weight!=None:
        import pdb
        pdb.set_trace()
        model.load_weights(weight)
        print "model of {} loaded".format(weight)

    return model

def getModel2(weight=None):
    from keras import backend as K
    K.set_image_dim_ordering('tf')
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import SGD, RMSprop

    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same',input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    # model.add(Convolution2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    print model.summary()
    if weight!=None:
        model.load_weights(weight)
        print "model of {} loaded".format(weight)

    return model



def create_dictionary_dl(lmbd, K=100, N=10000, dir_mnist='save_exp_cifar'):
    import os.path as osp
    dict = unpickle("/data/home/dujw/darse/save_exp/cifar/cifar_feature.pkl")
    fname = osp.join(dir_mnist, "D_mnist_K{}_lmbd{}.npy".format(K, lmbd))
    if osp.exists(fname):
        D = np.load(fname)
    else:
        from sklearn.decomposition import DictionaryLearning
        X = dict['feature']
        dl = DictionaryLearning(K, alpha=lmbd*N, fit_algorithm='cd',n_jobs=-1, verbose=1)
        dl.fit(X)
        D = dl.components_.reshape(K, -1)
        np.save(fname, D)
    return D 
if __name__ == '__main__':
    get_truth(100)
    # createFeature()

