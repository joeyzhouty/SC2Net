import tensorflow as tf


def soft_thresholding(x, theta):
    with tf.name_scope("Soft_thresholding"):
        return tf.nn.relu(x-theta) - tf.nn.relu(-x-theta)


def array_depl(ax, x0, x1, c='r', ls='--'):
    ax.arrow(x0[0], x0[1], (x1-x0)[0], (x1-x0)[1],
             color=c, linestyle=ls, length_includes_head=True)
