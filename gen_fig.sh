#!/usr/bin/sh

python mk_curve.py --exp sparse.05 --save curve_sparse005_large -x 250 -y 200 --noshow
python mk_curve.py --exp sparse.05 --save curve_sparse005_seaborn -x 250 -y 200 --seaborn --noshow

python mk_curve.py --exp sparse.2 --save curve_sparse02_large -x 500 -y 200 --noshow
python mk_curve.py --exp sparse.2 --save curve_sparse02_seaborn -x 500 -y 200 --seaborn --noshow

python mk_curve.py --exp mnist_100_05  -x 500 -y 1500 --eps 1e-2 --save curve_mnist_large --noshow
python mk_curve.py --exp mnist_100_05  -x 500 -y 1500 --eps 1e-2 --seaborn --save curve_mnist_seaborn --noshow

python mk_curve.py --exp images  -x 500 --save curve_images_large --noshow
python mk_curve.py --exp images  -x 500 --seaborn --save curve_images_seaborn --noshow

python mk_curve.py --exp adverse -x 1000 -y 100 --rm lfista linear -y 1 --save curve_adverse_large --noshow
python mk_curve.py --exp adverse -x 1000 -y 100 --rm lfista linear -y 1 --seaborn --save curve_adverse_seaborn --noshow

python plot_adverse_dictionary.py --save dictionary --noshow