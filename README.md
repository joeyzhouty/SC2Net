#SC2Net
Source codes for 

"SC2Net: Sparse LSTMs for Sparse Coding" (AAAI 2018)

"An End-to-end Sparse coding" Presented at the ICML 2017 Workshop on Principled Approaches
to Deep Learning, Sydney, Australia, 2017

The keras based implementation code for SLSTM can be found in Lcod/sparse_lstm.py 

The tensorflow based implementaion code for baseline algorithms lista and lfista can be found in Lcod/lista_network.py Lcod/lfista_network.py. We modify the loss functions of LISTA and FLISTA (parts of codes from [Thomas Moreau github projects](https://github.com/tomMoral/AdaptiveOptim)) to fit the SC2Net framework.

More complete code with the Tensorflow implementation will be available soon....



### Requirement
 * numpy 1.10+
 * matplotlib 1.8+
 * tensorflow 1.0+
 * keras 2.0.0+
 * python 2.7
 * scikit-learn 1.16+
 

### Usage
To use sparse_lstm independently 

    from Lcod.sparse_lstm import Sparse_LSTM_wo_0_Gate_v2
    slstm = Sparse_LSTM_wo_O_Gate_v2(z_dim)
    inputs = Input(shape=(iterations, x_dim), name='input')
    output = slstm(inputs)
    
To use lista lfista independently

    from Lcod.lista_network import LIstaNetwork
    from Lcod.lfista_network import LFistaNetwork
Different from the sparse_lstm, lista and lfista require pre-learned dictionary to initialize weights, the dictionary is pre learned by sklearn.decomposition.DictionaryLearning.  

    lis = LIstaNetwork(D, n_layers=10, shared=True,supervised=False, log_lvl=log_lvl,gpu_usage=gpu_usage, exp_dir=NAME_EXP,feed_lmbd=lmbd,Zpflag = False)
    lfis = LFistaNetwork(D, n_layers=10, shared=True, supervised=False, log_lvl=log_lvl,gpu_usage=gpu_usage,exp_dir=NAME_EXP,feed_lmbd=lmbd,Zpflag = False)
    #D is the pre-learned dictionary
    
    lis.train(batch_provider=pb, max_iter=20, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
    lfis.train(batch_provider=pb, max_iter=20, steps=steps, feed_val=feed_val_lista, reg_cost=8, tol=1e-8,lr_init=lr_init)
    #pb is the dataset generator

To run mnist & cifar reconstruction test



    python figures.py --data mnist --lmbd .1 -K 100 --save_dir mnist
    python figures.py --data cifar --lmbd .1 -K 100 --save_dir cifar
    #--data determines the dataset to be trained, --K determines the output sparce code length.

To train customized dataset, you need to write a dataset generator class which is similar to Lcod/mnist_problem_generator.py, the generator should contain a class to generate data by batch and a method to generate the pre-learned dictionary. 

    from Lcod.mnist_problem_generator import MnistProblemGenerator
    from Lcod.mnist_problem_generator import create_dictionary_dl
    D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=save_dir)
    pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=save_dir, seed=42242) 
### Result
####Mnist 
Reconstructed images on MNIST: The more black
is, the lower error is
![Mnist](pic/mnist.png?raw=true)



####Cifar
Reconstructed images on CIFAR-10
![Cifar](pic/cifar.png?raw=true)

