try:
    import sys
    sys.path.remove("/usr/lib/python3/dist-packages")
except ValueError:
    pass

import os
import json
import numpy as np
import matplotlib.pyplot as plt

if os.path.exists(os.path.join('/proc', 'acpi', 'bbswitch')):
    # assert that the graphic card is on if bbswitch is detected
    import os
    assert 'BUMBLEBEE_SOCKET' in os.environ.keys()

from Lcod.lista_network import LIstaNetwork
from Lcod.lfista_network import LFistaNetwork
from Lcod.facto_network import FactoNetwork
from Lcod.ista_tf import IstaTF
from Lcod.fista_tf import FistaTF
from Lcod.linear_network import LinearNetwork


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


def get_problem(config):

    # retrieve the parameter of the problem
    dataset = config['data']
    batch_size, lmbd = config['batch_size'], config['lmbd']
    seed = config.get('seed')

    # Setup the training constant and a test set
    if dataset == 'artificial':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from Lcod.simple_problem_generator import create_dictionary

        # retrieve specific parameters for the problem
        K, p, rho = config['K'], config['p'], config['rho']
        seed_D, corr = config.get('seed_D'), config.get('corr', 0)
        D = create_dictionary(K, p, seed=seed_D)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=seed)
    elif dataset == 'adverse':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from data_handlers.dictionaries import create_adversarial_dictionary

        # retrieve specific parameters for the problem
        K, p, rho = config['K'], config['p'], config['rho']
        seed_D, corr = config.get('seed_D'), config.get('corr', 0)
        D = create_adversarial_dictionary(K, p, seed=seed_D)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=seed)
    elif dataset == 'mnist':
        from Lcod.mnist_problem_generator import MnistProblemGenerator
        from Lcod.mnist_problem_generator import create_dictionary_dl
        K, save_dir = config['K'], config['save_dir']
        D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=save_dir)
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=save_dir, seed=seed)
    elif dataset == 'images':
        from Lcod.image_problem_generator import ImageProblemGenerator
        from Lcod.image_problem_generator import create_dictionary_haar
        p = config['p']
        D = create_dictionary_haar(p)
        pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                                   seed=seed)
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
    with open(file) as f:
        runs = json.load(f)
    run_exps = runs['run_exps']
    for exp in run_exps:
        for k, v in exp.items():
            if type(v) is str:
                exp[k] = runs[v]
    return run_exps


def load_exp(exp_name):
    exp_dir = os.path.join("exps", exp_name)
    file = os.path.join(exp_dir, "config.json")
    with open(file) as f:
        exps = json.load(f)
    exps['save_dir'] = exp_dir
    run_exps = parse_runfile(os.path.join(exp_dir, exps["runfile"]))
    pb, D = get_problem(exps)
    return D, pb, run_exps, exps


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Experiments for Adaopt paper')
    parser.add_argument('--exp', type=str, default="default",
                        help='If set, the experiments will be saved in the '
                             'specified directory')
    parser.add_argument('--gpu', type=float, default=.95,
                        help='Ratio of usage of the gpu for this launch')
    parser.add_argument('--debug', '-d', type=int, default=20,
                        help='Logging level, default is INFO, '
                             'set 10 for DEBUG')
    parser.add_argument('--linear', '-l', action='store_true',
                        help='recompute the linear layer model')
    args = parser.parse_args()

    # General Setup and constants
    # Experiment metadata
    exp_name = args.exp   # name of the experiment
    gpu_usage = args.gpu  # GPU memory allocated to this process
    log_lvl = args.debug  # verbosity -- INFO

    # Setup saving variables
    _assert_exist('exps')
    save_dir = _assert_exist('exps', exp_name)
    _assert_exist(save_dir, 'ckpt')
    save_curve = os.path.join(save_dir, "curve_cost.npy")

    D, pb, run_exps, exps = load_exp(exp_name)

    # Load variable value from json
    lmbd, lr_init, reg_scale, steps, warm_params = [
        exps.get(k) for k in ['lmbd', 'lr_init', 'reg_scale',
                              'steps', 'warm_params']]
    layer_lvl = [v['n_layers'] for v in run_exps]

    sig_test, z0_test, zs_test, _ = pb.get_test(exps['N_test'])
    sig_val, z0_val, zs_val, _ = pb.get_batch(exps['N_val'])
    C0 = pb.lasso_cost(zs_test, sig_test)

    # Compute the validation cost
    fista = FistaTF(D, gpu_usage=gpu_usage)
    fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                   max_iter=10000, tol=1e-8*C0)
    fista.terminate()
    c_val = fista.train_cost[-1]

    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd}
    feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}

    # Reload past experiment points
    networks = {}
    try:
        curve_cost = np.load(save_curve).take(0)
    except FileNotFoundError:
        curve_cost = {'lista': 2*C0*np.ones(len(layer_lvl)),
                      'lfista': 2*C0*np.ones(len(layer_lvl)),
                      'facto': 2*C0*np.ones(len(layer_lvl)),
                      }
        np.save(save_curve, curve_cost)

    # compute iterative algorihms train_cost if needed
    for m, o in [('ista', IstaTF), ('fista', FistaTF)]:
        if m not in curve_cost.keys():
            alg = o(D, gpu_usage=gpu_usage)
            alg.optimize(X=sig_test, lmbd=lmbd, Z=zs_test,
                         max_iter=10000, tol=1e-8*C0)
            curve_cost[m] = alg.train_cost
            cc = np.load(save_curve).take(0)
            cc[m] = curve_cost[m]
            np.save(save_curve, cc)
            alg.terminate()
    if 'linear' not in curve_cost.keys() or args.linear:
        # Compute the first layer of linear models
        network = LinearNetwork(D, 1, gpu_usage=gpu_usage, exp_dir=exp_name)
        network.train(pb, max_iter=500, steps=steps, feed_val=feed_val,
                      reg_cost=8, tol=1e-8, lr_init=1e-1)
        linear = IstaTF(D, gpu_usage=gpu_usage)
        linear.optimize(X=sig_test, lmbd=lmbd, Z=network.output(**feed_test),
                        max_iter=10000, tol=1e-8*C0)
        network.save(os.path.join(save_dir, 'ckpt', 'linear'))
        network.terminate()
        linear.terminate()
        curve_cost['linear'] = linear.train_cost

    cc = np.load(save_curve).take(0)
    cc['linear'] = curve_cost['linear']
    np.save(save_curve, cc)

    # Run the experiments
    models = [('lista', LIstaNetwork), ('lfista', LFistaNetwork),
              ('facto', FactoNetwork)]
    wp = {}
    for m, _ in models:
        wp[m] = []

    c_star = min(min(curve_cost['ista']), min(curve_cost['fista']))
    for i, expe in enumerate(run_exps):
        n_layers = expe['n_layers']
        for model, obj in models:
            key = '{}_{}'.format(model, n_layers)
            if expe[model] > 0:
                if key not in networks.keys():
                    network = networks[key] = obj(
                        D, n_layers=n_layers, shared=False, log_lvl=log_lvl,
                        gpu_usage=gpu_usage, warm_param=wp[model],
                        exp_dir=exp_name, reg_scale=reg_scale)
                    # C0 = network.cost(**feed_test)
                    # try:
                    #     if i == 0:
                    #         assert np.isclose(C0, curve_cost['ista'][1])
                    #     else:
                    #         assert curve_cost[model][i-1] > C0 > c_star
                    # except AssertionError as e:
                    #     print(e)
                    #     import IPython
                    #     IPython.embed()
                else:
                    network = networks[key]
                    # network.reset()
                try:
                    network.train(
                        pb, feed_val, max_iter=expe[model], steps=steps,
                        reg_cost=8, tol=1e-8,
                        # lr_init=lr_init/n_layers)
                        lr_init=lr_init)
                    import IPython
                    IPython.embed()
                except KeyboardInterrupt:
                    import IPython
                    IPython.embed()
                if warm_params:
                    wp[model] = network.export_param()

                try:
                    curve_cost[model][i] = network.cost(**feed_test)
                except IndexError:
                    cc = curve_cost[model]
                    curve_cost[model] = 2*C0*np.ones(len(layer_lvl))
                    curve_cost[model][:len(cc)] = cc

                cc = np.load(save_curve).take(0)
                cc[model][i] = curve_cost[model][i]
                np.save(save_curve, cc)
                try:
                    np.save(os.path.join(
                        save_dir, 'ckpt', '{}_weights'.format(key)),
                            [[]] + network.export_param())
                except ValueError:
                    print("Error in param saving for model {}".format(key))
                    raise
                network.save(os.path.join(save_dir, 'ckpt', key))
                network.terminate()
            elif warm_params and key in networks.keys():
                wp[model] = networks[key].export_param()
            elif warm_params:
                try:
                    wp[model] = np.load(os.path.join(
                        save_dir, 'ckpt', '{}_weights.npy'.format(key)))[1:]
                except FileNotFoundError:
                    pass

    curve_cost['lfista'][0] = curve_cost['lista'][0]
    cc = np.load(save_curve).take(0)
    cc[model][i] = curve_cost[model][i]
    np.save(save_curve, cc)

    import IPython
    IPython.embed()
