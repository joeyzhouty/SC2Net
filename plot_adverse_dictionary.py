import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_handlers.dictionaries import create_adversarial_dictionary
from data_handlers.dictionaries import create_gaussian_dictionary
sns.reset_orig()
w, h = 7.0, 4.0
mpl.rcParams['figure.figsize'] = [w, h]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Plot dictionary')
    parser.add_argument('--save', type=str, default=None,
                        help='')
    parser.add_argument('--gaussian', action="store_true",
                        help='Create a figure with a guassian dictionary')
    parser.add_argument('--noshow', action="store_true",
                        help='Do not display the graph')
    args = parser.parse_args()

    if args.gaussian:
        K, p = 100, 64
        D = create_gaussian_dictionary(K, p, seed=290890)
    else:
        K, p = 256, 50
        D = create_adversarial_dictionary(K, p, seed=420829)
    cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, dark=.2, reverse=True,
                                 as_cmap=True)
    fig = plt.figure("Dictionary")
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=.12, top=.99, right=.99)

    ax.imshow(D.T, aspect=K/p*h/w, cmap=cmap, interpolation="none")
    ax.set_xlabel("$m$", fontsize='x-large')
    ax.set_ylabel(" $n$", fontsize='x-large')
    ax.set_xticks([])
    ax.set_yticks([])

    if args.save is not None:
        plt.savefig("../Loptim/images/{}.pdf".format(args.save), dpi=150)
    if not args.noshow:
        plt.show()
