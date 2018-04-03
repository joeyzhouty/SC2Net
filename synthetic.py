import os
import numpy as np
import matplotlib.pyplot as plt
import errno
import pandas as pd

if os.path.exists(os.path.join('/proc', 'acpi', 'bbswitch')):
    # assert that the graphic card is on if bbswitch is detected
    import os
    assert 'BUMBLEBEE_SOCKET' in os.environ.keys()

try:
    import sys
    sys.path.remove("/usr/lib/python3/dist-packages")
except ValueError:
    pass

from Lcod.lista_network import LIstaNetwork
from Lcod.lfista_network import LFistaNetwork
from Lcod.facto_network import FactoNetwork
from Lcod.ista_tf import IstaTF
from Lcod.fista_tf import FistaTF
from Lcod.linear_network import LinearNetwork
from Lcod.Darse import DarseNetwork
from sklearn.linear_model import LogisticRegression as lgr
from sklearn.metrics import accuracy_score

def mk_curve(curve_cost, max_iter=1000, eps=1e-6):
    # Plot the layer curve
    c_star = min(curve_cost['ista'][-1], curve_cost['fista'][-1])-eps

    fig = plt.figure('Curve layer')
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=.12, top=.99)

    y_max = 0
    sym = 10
    max_iter = min(max_iter, len(curve_cost['ista']))
    layer_lvl = [1, 2, 4, 7, 12, 21, 35, 59, 100]

    for model, name, style in [('lista', 'L-ISTA', 'bo-'),
                               ('lfista', 'L-FISTA', 'c*-'),
                               ('facto', 'FacNet', 'rd-')]:
        cc = np.maximum(curve_cost[model]-c_star, eps)
        y_max = max(y_max, cc[0])
        ax.loglog(layer_lvl, cc, style, label=name)

    for model, name, style in [('ista', 'ISTA', 'g-'),
                               ('fista', 'FISTA', 'ys-'),
                               ('linear', 'Linear', 'g--o')]:
        cc = np.maximum(curve_cost[model]-c_star, eps)
        y_max = max(y_max, cc[0])
        iters = min(max_iter, len(cc))
        makers = np.unique((10**np.arange(0, np.log10(iters-1), 2/9)
                            ).astype(int))-1
        t = range(1, len(cc))
        ax.loglog(t, cc[1:], style,
                  # markevery=makers,
                  label=name)

    ax.hlines([eps], 1, max_iter, 'k', '--')

    ax.legend(fontsize='x-large', ncol=2)
    ax.set_xlim((1, max_iter))
    ax.set_ylim((eps/2, sym*y_max))
    ax.set_xlabel('# iteration/layers k', fontsize='x-large')
    ax.set_ylabel('Cost function $F(z) - F(z^*)$', fontsize='x-large')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)


def get_problem(dataset, K, p, lmbd, rho, batch_size, save_dir):
    # Setup the training constant and a test set
    if dataset == 'artificial':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from Lcod.simple_problem_generator import create_dictionary
        D = create_dictionary(K, p, seed=290890)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=422742)
    elif dataset == 'mnist':
        from Lcod.mnist_problem_generator import MnistProblemGenerator
        from Lcod.mnist_problem_generator import create_dictionary_dl
        D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=save_dir)
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=save_dir, seed=42242)
    elif dataset == 'images':
        from Lcod.image_problem_generator import ImageProblemGenerator
        from Lcod.image_problem_generator import create_dictionary_haar
        p = int(np.sqrt(p))
        D = create_dictionary_haar(p, wavelet='haar')
        pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                                   seed=1234)
    else:
        raise NameError("dataset {} not reconized by the script"
                        "".format(dataset))
    return pb, D


def _assert_exist(*args):
    """create a directory if it does not exist."""
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def parse_runfile(file):
    import json
    with open(file) as f:
        exps = json.load(f)
    run_exps = exps['run_exps']
    for exp in run_exps:
        for k, v in exp.items():
            if type(v) is str:
                exp[k] = exps[v]
    return run_exps

def save_csv(savename,columns,data):
    data = np.array(data)
    data = np.swapaxes(data,0,1)
    save_file = pd.DataFrame(data,columns=columns)
    save_file.to_csv("csv/features/"+savename+".csv",index=False)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Experiments for Adaopt paper')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='If set, the experiments will be saved in the '
                             'specified directory')
    parser.add_argument('--gpu', type=float, default=1.,
                        help='Ratio of usage of the gpu for this launch')
    parser.add_argument('--lmbd', type=float, default=.01,
                        help='Lambda used for the experiments, control the '
                             'regularisation level')
    parser.add_argument('--rho', type=float, default=.05,
                        help='Rho used for the experiments, control the '
                             'sparsity level')
    parser.add_argument('--debug', '-d', type=int, default=20,
                        help='Logging level, default is INFO, '
                             'set 10 for DEBUG')
    parser.add_argument('--data', type=str, default='artificial',
                        help='Dataset to run the experiments. Can be one of '
                             '["artificial", "images", "mnist"]')
    parser.add_argument('-K', type=int, default=100,
                        help='Number of dictionary elements used.')
    args = parser.parse_args()

    # General Setup and constants
    # Experiment metadata
    save_exp = args.save_dir is not None
    NAME_EXP = args.save_dir if args.save_dir else 'default'
    gpu_usage = args.gpu  # GPU memory allocated to this process
    log_lvl = args.debug  # verbosity -- INFO
    dataset = args.data   # dataset to run the experiment

    # Constants
    K = args.K             # Number of dictionary atoms
    N_test = 10000         # Number of test sample
    N_val = 1000           # Number of test sample

    lmbd = args.lmbd        # Regularisation level
    rho = args.rho          # Sparsity level
    corr = 0                # Correlation level for the coefficients
    eps = 1e-6              # Resolution for the optimization problem
    reg_scale = 1           # scaling of the unitary penalization

    # Extra network params
    warm_params = True      # Reuse the parameters from smaller network
    lr_init = 5e-2          # Initial learning rate for the gradient descent
    lr_init = 1e-1          # Initial learning rate for the gradient descent
    lr_fn = 1e-3            # Initial learning rate for GD in FacNet
    # steps = 100             # Number of steps fo GD between validation
    steps = 30
    batch_size = 300        # Size of the batch for the training

    # Setup the experiment plan
    it_lista = it_lfista = 0
    it_facto = 600
    run_exps = [
        {'n_layers': 1,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 2,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 4,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 7,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 12,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 21,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 35,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 59,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
        {'n_layers': 100,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                ]
    layer_lvl = [v['n_layers'] for v in run_exps]

    # Setup saving variables
    _assert_exist('save_exp')
    save_dir = _assert_exist('save_exp', NAME_EXP)
    _assert_exist(save_dir, 'ckpt')
    save_curve = os.path.join(save_dir, "curve_cost.npy")

    # Setup the training constant and a test set
    if dataset == 'artificial':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from Lcod.simple_problem_generator import create_dictionary
        p = 64                 # Dimension of the data
        D = create_dictionary(K, p, seed=290890)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=422742)
    elif dataset == 'mnist':
        from Lcod.mnist_problem_generator import MnistProblemGenerator
        from Lcod.mnist_problem_generator import create_dictionary_dl
        D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=save_dir)
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=save_dir, seed=42242)
    elif dataset == 'images':
        from Lcod.image_problem_generator import ImageProblemGenerator
        from Lcod.image_problem_generator import create_dictionary_haar
        p = 8
        reg_scale = 1e-4
        D = create_dictionary_haar(p)
        pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                                   data_dir='data/VOC', seed=1234)
    elif dataset == 'cifar':
        from Lcod.cifar_generator import CifarProblemGenerator
        from Lcod.cifar_generator import create_dictionary_dl
        p = 8
        reg_scale = 1e-4
        D = create_dictionary_dl(lmbd,K)
        # D = np.random.random([1024,3072])
        pb = CifarProblemGenerator(D, lmbd, batch_size=batch_size,
                                    seed=1234)
    elif dataset == 'synthetic':
        from Lcod.synthetic_generator import SyntheticProblemGenerator
        from Lcod.synthetic_generator import create_dictionary_dl
        p = 8
        reg_scale = 1e-4
        # D = create_dictionary_dl(lmbd=lmbd,d=2,m=100,n=20)

        pb = SyntheticProblemGenerator(lmbd=lmbd,d=2,m=100,n=20,case=1)
        # D = np.swapaxes(pb.phi,0,1)
        D = np.transpose(pb.phi)
        D = D + 0.02*np.random.normal(size=[100,20])
        from Lcod.utils import mul_coher
        # while(mul_coher(D)<0.3):
            # D = np.random.normal(size=[100,20])
        # D = np.random.random([100,20])
    else:
        raise NameError("dataset {} not reconized by the script"
                        "".format(dataset))

    sig_test, z0_test, zs_test, _ = pb.get_test(N_test)
    sig_val, z0_val, zs_val, lb,_ = pb.get_batch(N_val)


    # Compute optimal values for validation/test sets using ISTA/FISTA
    ista = IstaTF(D, gpu_usage=gpu_usage)
    # z_curr =ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,max_iter=1000)
    c_val = 1e-10

    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd}
    feed_val = {"Z": zs_val, "X": sig_val, "label":lb,"lmbd": lmbd, "c_val": c_val-1e-10}

    feed_val_lista = {"Z": zs_val, "Zp": zs_val, "X": sig_val, "label":lb,"lmbd": lmbd, "c_val": c_val-1e-10}

    #initiate all models 
    ista = IstaTF(D, gpu_usage=gpu_usage)
    fista = FistaTF(D, gpu_usage=gpu_usage)
    lfis = LFistaNetwork(D, n_layers=10, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage,exp_dir=NAME_EXP,Zpflag=True)
    lis = LIstaNetwork(D, n_layers=10, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage, exp_dir=NAME_EXP,Zpflag = True)
    

    #compute ds 
    ds = [1,2,4,6,8]
    # ds = [2]
    # lmbds = []
    columns =["d","ISTA","FISTA","LISTA-SC2Net","LFISTA-SC2Net"]
    cost = [ds,[],[],[],[]]
    acry = [ds,[],[],[],[]]
    from utils import mul_coher
    mul = [ds,[],[],[],[]]
    
    import numpy as np
    # acry[1] = np.random.random([7])
    # acry[2] = np.random.random([7])
    # acry[3] = np.random.random([7])
    # save_csv("lmbds_acry",columns,acry)
    for ddd in ds:
        lmbd= 0.1 
        feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}
        feed_val_lista = {"Z": zs_val, "Zp": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}
        # pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                       # dir_mnist=save_dir, seed=42242)
        pb = SyntheticProblemGenerator(lmbd=lmbd,d=ddd,m=100,n=20,case =1)
        D = np.transpose(pb.phi)
        D = D + 0.02*np.random.normal(size=[100,20])

        lfis = LFistaNetwork(D, n_layers=7, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage,exp_dir=NAME_EXP,feed_lmbd=lmbd,Zpflag = False)
        lis = LIstaNetwork(D, n_layers=7, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage, exp_dir=NAME_EXP,feed_lmbd=lmbd,Zpflag = False)
        # ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  # max_iter=20)
        # cost[1].append(ista.train_cost[-1])
        # acry[1].append(ista.test_loss(pb,lmbd=lmbd))

        # fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,max_iter=20)
        # cost[2].append(fista.train_cost[-1])
        # acry[2].append(fista.test_loss(pb,lmbd=lmbd))

        mul[1].append(mul_coher(np.transpose(pb.phi)))
        mul[2].append(mul_coher(D))
        lis.train(batch_provider=pb, max_iter=100, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
        # cost[3].append(lis.cost(**feed_val))
        # acry[3].append(lis.test_loss(pb,lmbd=lmbd))
        mul[3].append(lis.test_mul(pb,lmbd=lmbd))

        lfis.train(batch_provider=pb, max_iter=100, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
        # cost[4].append(lfis.cost(**feed_val))
        # acry[4].append(lfis.test_loss(pb,lmbd=lmbd))
        mul[4].append(lfis.test_mul(pb,lmbd=lmbd))

    # save_csv("synthetic_lmbds_acry",columns,acry)
    # save_csv("synthetic_lmbds_cost",columns,cost)
    save_csv("synthetic_mul_case1",columns,mul)


    import pdb
    pdb.set_trace()

    #compute lmbds 
    lmbds = [0.01,0.1,1]
    # lmbds = []
    columns =["lmbd","ISTA","FISTA","LISTA-SC2Net","LFISTA-SC2Net"]
    cost = [lmbds,[],[],[],[]]
    acry = [lmbds,[],[],[],[]]
    from utils import mul_coher
    mul = [lmbds,[],[],[],[]]
    
    import numpy as np
    # acry[1] = np.random.random([7])
    # acry[2] = np.random.random([7])
    # acry[3] = np.random.random([7])
    # save_csv("lmbds_acry",columns,acry)
    for lmbd in lmbds:
        feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}
        feed_val_lista = {"Z": zs_val, "Zp": z_curr, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}
        # pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                       # dir_mnist=save_dir, seed=42242)
        lfis = LFistaNetwork(D, n_layers=10, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage,exp_dir=NAME_EXP,feed_lmbd=lmbd,Zpflag = False)
        lis = LIstaNetwork(D, n_layers=10, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage, exp_dir=NAME_EXP,feed_lmbd=lmbd,Zpflag = False)
        ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=20)
        cost[1].append(ista.train_cost[-1])
        acry[1].append(ista.test_loss(pb,lmbd=lmbd))

        fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,max_iter=20)
        cost[2].append(fista.train_cost[-1])
        acry[2].append(fista.test_loss(pb,lmbd=lmbd))

        mul[1].append(mul_coher(np.transpose(pb.phi)))
        mul[2].append(mul_coher(D))
        lis.train(batch_provider=pb, max_iter=100, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
        cost[3].append(lis.cost(**feed_val))
        acry[3].append(lis.test_loss(pb,lmbd=lmbd))
        mul[3].append(lis.test_mul(pb,lmbd=lmbd))

        lfis.train(batch_provider=pb, max_iter=100, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
        cost[4].append(lfis.cost(**feed_val))
        acry[4].append(lfis.test_loss(pb,lmbd=lmbd))
        mul[4].append(lfis.test_mul(pb,lmbd=lmbd))

    # save_csv("synthetic_lmbds_acry",columns,acry)
    # save_csv("synthetic_lmbds_cost",columns,cost)
    save_csv("synthetic_mul",columns,mul)



# computer layers
    # layers = [4,7,12,21,35,59]
    layers = []
    # columns =["layers","ista","fista","lista","lfista"]
    columns =["layers","ISTA","FISTA","LISTA-SC2Net","LFISTA-SC2Net"]
    cost = [layers,[],[],[],[]]
    acry = [layers,[],[],[],[]]
    lmbd = 0.1
    for ly in layers:
        feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}
        feed_val_lista = {"Z": zs_val, "Zp": z_curr, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                       dir_mnist=save_dir, seed=42242)

        lfis = LFistaNetwork(D, n_layers=3, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage,exp_dir=NAME_EXP,Zpflag = False)
        lis = LIstaNetwork(D, n_layers=3, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage, exp_dir=NAME_EXP,Zpflag = False)


        ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=ly)
        cost[1].append(ista.train_cost[-1])
        acry[1].append(ista.test_accuracy(pb,lmbd=lmbd,max_iter=ly))

        fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,max_iter=ly)
        cost[2].append(fista.train_cost[-1])
        acry[2].append(fista.test_accuracy(pb,lmbd=lmbd,max_iter=ly))


        lis.train(batch_provider=pb, max_iter=ly, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
        cost[3].append(lis.cost(**feed_val))
        acry[3].append(lis.test_accuracy(pb,lmbd=lmbd))

        lfis.train(batch_provider=pb, max_iter=ly, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
        cost[4].append(lfis.cost(**feed_val))
        acry[4].append(lfis.test_accuracy(pb,lmbd=lmbd))
    save_csv("mnist_layers_acry",columns,acry)
    save_csv("mnist_layers_cost",columns,cost)
    import pdb
    pdb.set_trace()



    #lfista network 
    layes=[1,2,3,5,7,10]
    report = {}
    cost=[]
    for i  in layes:
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                       dir_mnist=save_dir, seed=42242)
        lfis = LFistaNetwork(D, n_layers=i, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage, warm_param=wp["lista"],exp_dir=NAME_EXP, reg_scale=reg_scale)
        lfis.train(batch_provider=pb, max_iter=20, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init/7)
        accry =lfis.test_accuracy(pb,lmbd)
        report["lfista_"+str(i)] = accry
        print "lfista accru",accry
    ## listanetwork
    for i in layes:
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                       dir_mnist=save_dir, seed=42242)
        lis = LIstaNetwork(D, n_layers=i, shared=True, log_lvl=log_lvl,gpu_usage=gpu_usage, exp_dir=NAME_EXP, reg_scale=reg_scale,Zpflag = False)
        lis.train(batch_provider=pb, max_iter=20, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init/7)
        accry =lis.test_accuracy(pb,lmbd)
        report["lista_"+str(i)] = accry
        print "lista accru",accry
    import pdb
    pdb.set_trace()



    #Darse train
    darse.train(batch_provider=pb,max_iter=20,steps=steps,feed_val=feed_val, reg_cost=8, tol=1e-8,lr_init=lr_init/2)
    np.save("darse_cost.npy",darse.cost(**feed_test))
    for i, expe in enumerate(run_exps):
        n_layers = expe['n_layers']
        for model, obj in models:
            key = '{}_{}'.format(model, n_layers)
            if expe[model] > 0:
                if key not in networks.keys():
                    network = networks[key] = obj(
                        D, n_layers=n_layers, shared=False, log_lvl=log_lvl,
                        gpu_usage=gpu_usage, warm_param=wp[model],
                        exp_dir=NAME_EXP, reg_scale=reg_scale)
                else:
                    network = networks[key]
                    # network.reset()
                network.train(
                    batch_provider=pb, max_iter=expe[model], steps=steps, feed_val=feed_val, reg_cost=8, tol=1e-8,
                    # lr_init=lr_init if 'facto' != model else lr_fn/n_layers,
                    lr_init=lr_init/n_layers)
                if warm_params:
                    wp[model] = network.export_param()

                curve_cost[model][i] = network.cost(**feed_test)
                np.save(save_curve, curve_cost)
                try:
                    np.save(os.path.join(
                        save_dir, 'ckpt', '{}_weights'.format(key)),
                            [[]] + network.export_param())
                except ValueError:
                    print("Error in param saving for model {}".format(key))
                    raise
                network.terminate()
            elif warm_params and key in networks.keys():
                wp[model] = networks[key].export_param()
            elif warm_params:
                try:
                    wp[model] = np.load(os.path.join(
                        save_dir, 'ckpt', '{}_weights'.format(key)))[1:]
                except IOError as e:
                    if e.errno == errno.ENOENT:
                        pass

    curve_cost['lfista'][0] = curve_cost['lista'][0]
    np.save(save_curve, curve_cost)

    if save_exp:
        save_value = dict(layer_lvl=layer_lvl, curve_cost=curve_cost, pb=pb)
        import pickle
        from datetime import datetime
        t = datetime.now()
        save_file = 'save_layer{0.day:02}{0.month:02}.pkl'.format(t)
        with open(os.path.join(save_dir, save_file), 'wb') as f:
            pickle.dump(save_value, f)

    import IPython
    IPython.embed()
