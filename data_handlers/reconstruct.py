#python scripts
__author__='Du Jiawei'
#Descrption:
import numpy as np
import gzip, cPickle
import sys
sys.path.append("/data/home/dujw/darse/Lcod")
def loadmnist(N):
    with gzip.open("/data/home/dujw/darse/data_handlers/mnist.pkl.gz",'rb') as f:
        train_set_mnist, valid_set_mnist, test_set_mnist = cPickle.load(f)
    train_set_mnist_img, train_set_mnist_label = train_set_mnist
    return train_set_mnist_img[:N],train_set_mnist_label[:N]
def mnist_preprocess(origi,lis,lfis,lstm):
    patch_size =28
    origi = origi.reshape(100,patch_size,patch_size)
    lis = lis.reshape(100,patch_size,patch_size) - origi
    lfis = lfis.reshape(100,patch_size,patch_size) -origi
    lstm = lstm.reshape(100,patch_size,patch_size) -origi

    lis = np.abs(lis)
    lfis =np.abs(lfis)
    lstm = np.abs(lstm)

    # origi = origi.astype(np.uint8)
    # lis = lis.astype(np.uint8)
    # lfis = lfis.astype(np.uint8)
    # lstm = lstm.astype(np.uint8)


    return origi,lis,lfis,lstm

def cifar_preprocess(origi,lis,lfis,lstm):
    patch_size =32
    origi = origi.reshape(100,3,patch_size,patch_size)
    lis = lis.reshape(100,3,patch_size,patch_size)
    lfis = lfis.reshape(100,3,patch_size,patch_size)
    lstm = lstm.reshape(100,3,patch_size,patch_size)*255
    lis = lis.astype(np.uint8)
    lfis = lfis.astype(np.uint8)
    lstm = lstm.astype(np.uint8)
    lis = np.abs(lis)
    lfis =np.abs(lfis)
    lstm = np.abs(lstm)

    origi = origi.swapaxes(3,1)
    lis = lis.swapaxes(3,1)
    lfis = lfis.swapaxes(3,1)
    lstm = lstm.swapaxes(3,1)
    origi = np.rot90(origi,3,(1,2))
    lis = np.rot90(lis,3,(1,2))
    lfis = np.rot90(lfis,3,(1,2))
    lstm = np.rot90(lstm,3,(1,2))
    return origi,lis,lfis,lstm
def reconstrcuct_drow(pb,lis=0,lfis=0,path="/data/home/dujw/darse/data_handlers/mnist/prediction_iter3_lmbd0.1.pkl"):
    aa=cPickle.load(open(path,"r"))
    aa=aa[1]
    aa=np.array(aa)
    origi = loadmnist(100)[0]
    # origi = pb.get_truth(100)
    lstm = aa 
    # lis = lstm
    # lfis = lstm
    origi,lis,lfis,lstm = mnist_preprocess(origi,lis,lfis,lstm)
    # origi,lis,lfis,lstm = cifar_preprocess(origi,lis,lfis,lstm)
    from utils import combineArray 
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(8,8))
    gs1 = gridspec.GridSpec(2,8)
    gs1.update(wspace=0.025, hspace=0.05)
    plt.axis('off')
    # fig.suptitle("Mnisi recontruction error map", fontsize=16)
    ax=plt.subplot(221)
    ax.set_title("Ground Truth")
    # plt.imshow(combineArray(origi))
    plt.imshow(combineArray(origi),cmap='gray')
    # plt.tight_layout()
    
    # plt.colorbar()
    plt.axis('off')
    ax = plt.subplot(222)
    ax.set_title("Lista")
    # plt.imshow(combineArray(lis))
    plt.imshow(combineArray(lis),cmap='gray')
    # plt.tight_layout()
    
    # plt.colorbar()

    plt.axis('off')
    ax = plt.subplot(223)
    ax.set_title("Lfista")
    # plt.imshow(combineArray(lfis))
    plt.imshow(combineArray(lfis),cmap='gray')
    # plt.tight_layout()
    
    # plt.colorbar()
    plt.axis('off')
    ax = plt.subplot(224)
    ax.set_title("SC2Net")
    # plt.imshow(combineArray(lstm)
    plt.imshow(combineArray(lstm),cmap='gray')
    # plt.tight_layout()
    plt.axis('off')
    # plt.colorbar()
    cbar_ax = plt.gcf().add_axes([0.91, 0.1, 0.03, 0.8])
    plt.colorbar(cax=cbar_ax)
    # plt.tight_layout()
    # plt.show() 
    import pdb
    pdb.set_trace()
    plt.savefig('destination_path.eps', format='eps')
if __name__ == '__main__':
    from cifar_generator import CifarProblemGenerator
    D = np.random.random([1024,3072])
    lmbd = 0.1
    pb = CifarProblemGenerator(D, lmbd, batch_size=10)
    reconstrcuct_drow(pb,path="prediction_iter3_lmbd0.100000.pkl")
