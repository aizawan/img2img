import os
import sys
import tensorflow as tf


def get_weight_and_bias(weight_shape, bias_shape, weight_init, bias_init):
    weight = tf.get_variable("weight", weight_shape, initializer=weight_init)
    bias = tf.get_variable("bias", bias_shape, initializer=bias_init)
    return weight, bias


def conv2d(incoming, num_filters, filter_size, stride=1, pad='SAME',
           activation=tf.identity,
           weight_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
           bias_init=tf.constant_initializer(0.0),
           reuse=False, name="conv2d"):

    input_shape = incoming.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse) as scope:
        weight, bias = get_weight_and_bias(
            weight_shape=[filter_size,
                          filter_size,
                          input_shape[-1],
                          num_filters],
            bias_shape=[num_filters],
            weight_init=weight_init,
            bias_init=bias_init)
        conved = tf.nn.conv2d(incoming, weight, [1, stride, stride, 1], pad)
        conved = tf.nn.bias_add(conved, bias)
    return activation(conved)


def reflection_pad_conv2d(
        incoming, num_filters, filter_size, stride=1,
        activation=tf.identity,
        weight_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
        bias_init=tf.constant_initializer(0.0),
        reuse=False, name="conv2d"):

    input_shape = incoming.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse) as scope:
        weight, bias = get_weight_and_bias(
            weight_shape=[filter_size,
                          filter_size,
                          input_shape[-1],
                          num_filters],
            bias_shape=[num_filters],
            weight_init=weight_init,
            bias_init=bias_init)

        padding = [[0, 0], [filter_size // 2, filter_size // 2],
                   [filter_size // 2, filter_size // 2], [0, 0]]
        pad_x = tf.pad(incoming, padding, 'REFLECT')

        conved = tf.nn.conv2d(
            pad_x, weight, [1, stride, stride, 1], padding='VALID')
        conved = tf.nn.bias_add(conved, bias)
    return activation(conved)


def compute_deconv_output_shape(input_shape, filter_shape, stride, padding):
    bs, in_h, in_w, in_c = input_shape
    k_h, k_w, out_c, in_c = filter_shape
    _, s_h, s_w, _ = stride

    if padding == 'VALID':
        out_h = in_h * s_h + max(k_h - s_h, 0)
        out_w = in_w * s_w + max(k_w - s_w, 0)
    elif padding == 'SAME':
        out_h = in_h * s_h
        out_w = in_w * s_w

    return (bs, out_h, out_w, out_c)


def deconv2d(incoming, num_filters, filter_size, stride=1, padding='SAME',
             activation=tf.identity,
             weight_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
             bias_init=tf.constant_initializer(0.0),
             reuse=False, name='deconv2d'):

    input_shape = incoming.get_shape().as_list()
    strides = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, input_shape[-1]]
    output_shape = compute_deconv_output_shape(
        input_shape, filter_shape, strides, padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        weight, bias = get_weight_and_bias(
            weight_shape=filter_shape, bias_shape=[num_filters],
            weight_init=weight_init, bias_init=bias_init)

        deconved = tf.nn.conv2d_transpose(
            incoming, weight, output_shape, strides, padding)
        deconved = tf.nn.bias_add(deconved, bias)
    return activation(deconved)


def instance_norm(incoming, epsilon=1e-5,
                  beta_init=tf.constant_initializer(0.0),
                  gamma_init=tf.random_normal_initializer(
                  mean=1.0, stddev=0.002),
                  reuse=False, name='instance_norm'):
    input_shape = incoming.get_shape().as_list()
    depth = input_shape[-1]
    with tf.variable_scope(name, reuse=reuse) as scope:
        beta = tf.get_variable(
            'beta', shape=depth, initializer=beta_init, trainable=True)
        gamma = tf.get_variable(
            'gamma', shape=depth, initializer=gamma_init, trainable=True)

        mean, variance = tf.nn.moments(incoming, axes=[1, 2], keep_dims=True)
        inv = tf.rsqrt(variance + epsilon)
        norm = (incoming - mean) * inv
        output = gamma * norm + beta
    return output


def gaussian_noise(incoming, phase_train, avg=0.0, std=1.0,
                   name='gaussian_noise'):

    with tf.name_scope(name):
        def _gaussian_noise(x):
            noise = tf.random_normal(
                shape=tf.shape(x), mean=avg, stddev=std, dtype=tf.float32)
            return x + noise
        
        output = tf.cond(phase_train,
                         lambda: _gaussian_noise(incoming),
                         lambda: tf.identity(incoming))
    return output


def leakly_relu(incoming, alpha=0.1, name='leakly_relu'):
    x = incoming
    with tf.name_scope(name) as scope:
        m_x = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        x -= alpha * m_x
    return x
