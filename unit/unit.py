import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.getcwd())
from ops.layers import conv2d, gaussian_noise
from ops.blocks import c7s1_k, dk, Rk, uk, Ck


def generator(inputs, phase_train,
              enc_reuse,
              dec_reuse,
              ltn_reuse,
              num_enc_res_block=3,
              num_shared_res_block=2,
              num_dec_res_block=3,
              name_encoder='encoder',
              name_shared_space='shared_latent_space',
              name_decoder='decoder'):

    out_c = inputs.get_shape().as_list()[-1]

    # Generator: one domain -> another domain
    with tf.variable_scope('generator'):
        # Encoder converts one domain image to shared latent variables.
        fmaps = encoder(
            inputs=inputs,
            num_enc_res_block=num_enc_res_block,
            phase_train=phase_train,
            reuse=enc_reuse,
            name_encoder=name_encoder)

        # Shared latent space using weight sharing.
        fmaps, latent_vars = shared_latent_space(
            inputs=fmaps,
            num_shared_res_block=num_shared_res_block,
            phase_train=phase_train,
            reuse=ltn_reuse,
            name_shared_space=name_shared_space)

        # Decoder reconstracts another domain image from shared latent space.
        reconstracted_img = decoder(
            inputs=fmaps,
            out_c=out_c,
            num_dec_res_block=num_dec_res_block,
            phase_train=phase_train,
            reuse=dec_reuse,
            name_decoder=name_decoder)

    return reconstracted_img, latent_vars


def encoder(inputs, num_enc_res_block, phase_train, reuse, name_encoder):
    """ encoder
        inputs: [bs, 256, 256, in_c]
    """
    with tf.variable_scope(name_encoder):
        e = c7s1_k(inputs, 32, reuse, name='c7s1_32') # [bs, 128, 128, 32]
        e = dk(e, 64, reuse, name='d64') #[bs, 64, 64, 64]
        e = dk(e, 128, reuse, name='d128') # [bs, 32, 32, 128]
        for i in range(num_enc_res_block):
            e = Rk(e, 128, reuse, name='R128_{}'.format(i + 1)) # [bs, 32, 32, 128]
    return e


def shared_latent_space(inputs, num_shared_res_block, phase_train, reuse,
                        name_shared_space):
    """ shared_latent_space
    """
    with tf.variable_scope(name_shared_space):
        e = inputs
        for i in range(num_shared_res_block):
            e = Rk(e, 128, reuse, name='R128_e_{}'.format(i + 1))
        ltn = gaussian_noise(e, phase_train, name='gaussian_noise')
        d = ltn
        for i in range(num_shared_res_block):
            d = Rk(d, 128, reuse, name='R128_d_{}'.format(i + 1))
    return d, ltn


def decoder(inputs, out_c, num_dec_res_block, phase_train, reuse, name_decoder):
    """ decoder
    """
    with tf.variable_scope(name_decoder):
        d = inputs
        for i in range(num_dec_res_block):
            d = Rk(d, 128, reuse, name='R128_{}'.format(i + 1))
        d = uk(d, 64, reuse, name='u64') # [bs, 64, 64, 64]
        d = uk(d, 32, reuse, name='u32') # [bs, 128, 128, 32]
        d = c7s1_k(d, out_c, reuse, activation=False, is_norm=False,
                   name='c7s1_{}'.format(out_c)) # [bs, 128, 128, out_c]
        d = tf.nn.tanh(d, name='output')
    return d


def discriminator(real_img, fake_img, buffer_img, name_discriminator):
    # fake img (generated img from generator) or real img?
    with tf.variable_scope('discriminator'):
        with tf.variable_scope(name_discriminator):
            D_real_logits = _discriminator(real_img, reuse=False)
            D_fake_logits = _discriminator(buffer_img, reuse=True)
            G_fake_logits = _discriminator(fake_img, reuse=True)
    return (D_real_logits, D_fake_logits, G_fake_logits)


def generator_loss(fake_logits, ls_loss, c=0.9, epsilon=1e-12):
    if ls_loss:
        return tf.reduce_mean(tf.squared_difference(fake_logits, c))
    else:
        fake_prob = tf.nn.sigmoid(fake_logits)
        return -tf.reduce_mean(tf.log(fake_prob + epsilon))


def discriminator_loss(real_logits, fake_logits, ls_loss, a=0.0, b=0.9,
                       epsilon=1e-12):
    if ls_loss:
        real_loss = tf.reduce_mean(tf.squared_difference(real_logits, b))
        fake_loss = tf.reduce_mean(tf.squared_difference(fake_logits, a))
        loss = (real_loss + fake_loss) * 0.5
    else:
        real_prob = tf.nn.sigmoid(real_logits)
        fake_prob = tf.nn.sigmoid(fake_logits)
        loss = -tf.reduce_mean(
            tf.log(real_prob + epsilon) + tf.log(1. - fake_prob + epsilon))
    return loss


def kl(mu):
    mu_2 = tf.pow(mu, 2)
    encoding_loss = tf.reduce_mean(mu_2)
    return encoding_loss


def reconstract_loss(input_a, input_b):
    return tf.reduce_mean(tf.abs(input_b - input_a))


def _make_optimizer(loss, variables, beta1, learning_rate, name='Adam'):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = learning_rate
    end_learning_rate = 0.0
    start_decay_step = 100000
    decay_steps = 100000
    beta1 = beta1
    learning_rate = (tf.where(
        tf.greater_equal(global_step, start_decay_step),
        tf.train.polynomial_decay(
            starter_learning_rate, global_step-start_decay_step,
            decay_steps, end_learning_rate, power=1.0),
            starter_learning_rate))

    learning_step = (tf.train.AdamOptimizer(
        learning_rate, beta1=beta1, name=name).minimize(
            loss, global_step=global_step, var_list=variables))

    return learning_step


def _train_op(loss, learning_rate, beta1, scope_name):
    params = tf.get_collection(
        key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    return _make_optimizer(
        loss, params, beta1, learning_rate, name=scope_name + '_Adam')


def train_op(opt_generator_loss, opt_discrimintor_loss,
             generator_scope, discriminator_scope,
             generator_lr, discriminator_lr, beta1):

    ops_generator = _train_op(
        loss=opt_generator_loss,
        learning_rate=generator_lr,
        scope_name=generator_scope,
        beta1=beta1)

    ops_discriminator = _train_op(
        loss=opt_discrimintor_loss,
        learning_rate=discriminator_lr,
        scope_name=discriminator_scope,
        beta1=beta1)

    with tf.control_dependencies([ops_generator, ops_discriminator]):
        opt_op = tf.no_op(name='optimizers')

    return opt_op


def _discriminator(inputs, reuse):
    d = Ck(inputs, 64, reuse, is_norm=False, name='C64') # [bs, 128, 128, 64]
    d = Ck(d, 128, reuse, is_norm=True, name='C128') # [bs, 64, 64, 128]
    d = Ck(d, 256, reuse, is_norm=True, name='C256') # [bs, 32, 32, 256]
    d = Ck(d, 512, reuse, is_norm=True, name='C512') # [bs, 16, 16, 512]
    d = conv2d(d, 1, 4, stride=1, reuse=reuse, name='last_conv')
    logits = d
    return logits
