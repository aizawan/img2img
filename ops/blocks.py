import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.getcwd())
from ops.layers import (conv2d, deconv2d, reflection_pad_conv2d, 
                        instance_norm, leakly_relu, gaussian_noise)


def c7s1_k(inputs, k, reuse, is_norm=True, activation=True, name='c7s1_k'):
    """ c7s1_k is a 7*7 Convolution-InstanceNorm-ReLU Layer
        with k filters and stride 1.
    """
    with tf.variable_scope(name, reuse=reuse):
        h = reflection_pad_conv2d(inputs, k, 7, stride=1, name='conv_ref_pad')
        if is_norm: h = instance_norm(h, name='instance_norm')
        if activation: h = tf.nn.relu(h, name='relu')
    return h


def dk(inputs, k, reuse, name='dk'):
    """ dk is a 3*3 Convolution-InstanceNorm-ReLU layer
        with k filters and stride 2.
    """
    with tf.variable_scope(name, reuse=reuse):
        h = conv2d(inputs, k, 3, stride=2, name='conv')
        h = instance_norm(h, name='instance_norm')
        h = tf.nn.relu(h, name='relu')
    return h


def Rk(inputs, k, reuse, name='Rk'):
    """ Rk is a residual block that contains two 3*3 convolutional layer
        with the same number of filters on both layer.
    """
    with tf.variable_scope(name, reuse=reuse):
        h = reflection_pad_conv2d(inputs, k, 3, stride=1, name='conv_ref_pad_1')
        h = instance_norm(h, name='instance_norm_1')
        h = tf.nn.relu(h, name='relu_1')
        h = reflection_pad_conv2d(h, k, 3, stride=1, name='conv_ref_pad_2')
        h = instance_norm(h, name='instance_norm_2')
        h = inputs + h
        return h


def uk(inputs, k, reuse, name='uk'):
    """ uk is a 3*3 fractional-strided-Convolution-InstanceNorm-ReLU layer
        with k filters and stride 1/2.
    """
    with tf.variable_scope(name, reuse=reuse):
        h = deconv2d(inputs, k, 3, stride=2, name='fsconv')
        h = instance_norm(h, name='instance_norm')
        h = tf.nn.relu(h, name='relu')
    return h


def Ck(inputs, k, reuse, is_norm=True, name='Ck'):
    """ Ck is a 4*4 Convolution-InstanceNorm-LeakyReLU layer
        with k filters and stride 2.
    """
    with tf.variable_scope(name, reuse=reuse):
        h = conv2d(inputs, k, 4, stride=2, name='conv')
        if is_norm: h = instance_norm(h, name='instance_norm')
        h = leakly_relu(h, 0.2, name='leakly_relu')
    return h