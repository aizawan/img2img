import os
import sys
import time
from datetime import datetime
import logging
import importlib
import numpy as np
import tensorflow as tf
from collections import deque
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from unit import inputs
from utils import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

# Basic arguments
flags.DEFINE_string('arch', 'model', 'Network architecure')
flags.DEFINE_string('outdir', 'output/unit', 'Output directory')

# Dataset arguments
flags.DEFINE_string('x_tfrecord',
    '/tmp/data/unit/x-train-1000.tfrecord', 'TFRecord path')
flags.DEFINE_string('y_tfrecord',
    '/tmp/data/unit/y-train-1000.tfrecord', 'TFRecord path')

# Training arguments
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('target_height', 256, 'Input height')
flags.DEFINE_integer('target_width', 256, 'Input width')
flags.DEFINE_integer('target_channel', 3, 'Input channel')
flags.DEFINE_integer('iteration', 100000, 'Number of training iteration')
flags.DEFINE_integer('num_threads', 8, 'Number of threads to read batches')
flags.DEFINE_integer('min_after_dequeue', 10, 'min_after_dequeue')
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_integer('snapshot', 20000, 'Snapshot')

# Hyperprameters
flags.DEFINE_integer('num_enc_res_block', 3, 'residual block')
flags.DEFINE_integer('num_shared_res_block', 2, 'residual block')
flags.DEFINE_integer('num_dec_res_block', 3, 'residual block')
flags.DEFINE_float('generator_lr', 1e-4, 'learning rate')
flags.DEFINE_float('discriminator_lr', 1e-4, 'learning rate')
flags.DEFINE_float('beta1', 0.5, 'adam beta1')
flags.DEFINE_float('lambda_gan', 10., 'lambda_gan') # lambda 0
flags.DEFINE_float('lambda_vae_kl', 0.1, 'lambda_vae_kl') # lambda 1
flags.DEFINE_float('lambda_rec', 100., 'lambda_rec') # lambda 2
flags.DEFINE_float('lambda_cycle_kl', 0.1, 'lambda_cycle_kl') # lambda 3
flags.DEFINE_float('lambda_cycle', 100., 'lambda_cycle') # lambda 4
flags.DEFINE_bool('ls_loss', True, 'Use least square loss')

# Set random seed
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __call__(self, x, y, n=1):
        if len(self.buffer) > self.buffer_size:
            return random.sample(self.buffer, n)
        else:
            self.buffer.append((x, y))
            return x, y

    def __len__(self):
        return len(self.buffer)


def train(model_dir, summary_dir):
    logging.info('Training {}'.format(FLAGS.arch))
    logging.info('FLAGS: {}'.format(FLAGS.__flags))

    graph = tf.Graph()
    with graph.as_default():
        if FLAGS.y_tfrecord is None:
            x_fn_queue = tf.train.string_input_producer([FLAGS.x_tfrecord])
            domain_x, domain_y = inputs.input_pair(
                fn_queue=x_fn_queue,
                target_height=FLAGS.target_height,
                target_width=FLAGS.target_width,
                target_channel=FLAGS.target_channel,
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_threads,
                min_after_dequeue=FLAGS.min_after_dequeue,
                shuffle=True)
        else:
            x_fn_queue = tf.train.string_input_producer([FLAGS.x_tfrecord])
            y_fn_queue = tf.train.string_input_producer([FLAGS.y_tfrecord])
            domain_x = inputs.input_nopair(
                fn_queue=x_fn_queue,
                target_height=FLAGS.target_height,
                target_width=FLAGS.target_width,
                target_channel=FLAGS.target_channel,
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_threads,
                min_after_dequeue=FLAGS.min_after_dequeue,
                shuffle=True)
            domain_y = inputs.input_nopair(
                fn_queue=y_fn_queue,
                target_height=FLAGS.target_height,
                target_width=FLAGS.target_width,
                target_channel=FLAGS.target_channel,
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_threads,
                min_after_dequeue=FLAGS.min_after_dequeue,
                shuffle=True)

        # Set placeholder
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # fake data to update discriminators
        buffer_x = tf.placeholder(
            tf.float32, shape=domain_x.get_shape().as_list(), name='buffer_x')
        buffer_y = tf.placeholder(
            tf.float32, shape=domain_y.get_shape().as_list(), name='buffer_y')

        # Import network model.
        model = importlib.import_module("unit.{}".format(FLAGS.arch))

        # Build generator G1: X -> Z -> Y
        xy_img, xy_z = model.generator(
            inputs=domain_x,
            phase_train=phase_train,
            enc_reuse=False,
            dec_reuse=False,
            ltn_reuse=False,
            num_enc_res_block=FLAGS.num_enc_res_block,
            num_shared_res_block=FLAGS.num_shared_res_block,
            num_dec_res_block=FLAGS.num_dec_res_block,
            name_encoder='E1',
            name_decoder='G2')

        # Build generator G2: Y -> Z -> X
        yx_img, yx_z = model.generator(
            inputs=domain_y,
            phase_train=phase_train,
            enc_reuse=False,
            dec_reuse=False,
            ltn_reuse=True,
            num_enc_res_block=FLAGS.num_enc_res_block,
            num_shared_res_block=FLAGS.num_shared_res_block,
            num_dec_res_block=FLAGS.num_dec_res_block,
            name_encoder='E2',
            name_decoder='G1')

        # Build generator G1: X -> Z -> X
        xx_img, xx_z = model.generator(
            inputs=domain_x,
            phase_train=phase_train,
            enc_reuse=True,
            dec_reuse=True,
            ltn_reuse=True,
            num_enc_res_block=FLAGS.num_enc_res_block,
            num_shared_res_block=FLAGS.num_shared_res_block,
            num_dec_res_block=FLAGS.num_dec_res_block,
            name_encoder='E1',
            name_decoder='G1')

        # Build generator G2: Y -> Z -> Y
        yy_img, yy_z = model.generator(
            inputs=domain_y,
            phase_train=phase_train,
            enc_reuse=True,
            dec_reuse=True,
            ltn_reuse=True,
            num_enc_res_block=FLAGS.num_enc_res_block,
            num_shared_res_block=FLAGS.num_shared_res_block,
            num_dec_res_block=FLAGS.num_dec_res_block,
            name_encoder='E2',
            name_decoder='G2')

        # Backward x: X -> Z -> Y -> Z -> X
        xyx_img, xyx_z = model.generator(
            inputs=xy_img,
            phase_train=phase_train,
            enc_reuse=True,
            dec_reuse=True,
            ltn_reuse=True,
            num_enc_res_block=FLAGS.num_enc_res_block,
            num_shared_res_block=FLAGS.num_shared_res_block,
            num_dec_res_block=FLAGS.num_dec_res_block,
            name_encoder='E2',
            name_decoder='G1')

        # Backward x: Y -> Z -> X -> Z -> Y
        yxy_img, yxy_z = model.generator(
            inputs=yx_img,
            phase_train=phase_train,
            enc_reuse=True,
            dec_reuse=True,
            ltn_reuse=True,
            num_enc_res_block=FLAGS.num_enc_res_block,
            num_shared_res_block=FLAGS.num_shared_res_block,
            num_dec_res_block=FLAGS.num_dec_res_block,
            name_encoder='E1',
            name_decoder='G2')

        # Build X Discriminator DX
        D_X_real_logits, D_X_fake_logits, G_X_fake_logits = model.discriminator(
            real_img=domain_x,
            fake_img=yx_img,
            buffer_img=buffer_x,
            name_discriminator='D_X')

        # Build Y Discriminator DY
        D_Y_real_logits, D_Y_fake_logits, G_Y_fake_logits = model.discriminator(
            real_img=domain_y,
            fake_img=xy_img,
            buffer_img=buffer_y,
            name_discriminator='D_Y')

        # Generator loss
        GX_ad_loss = model.generator_loss(
            D_X_fake_logits, ls_loss=FLAGS.ls_loss)
        GY_ad_loss = model.generator_loss(
            D_Y_fake_logits, ls_loss=FLAGS.ls_loss)
        generator_loss = FLAGS.lambda_gan * (GX_ad_loss + GY_ad_loss)

        # Discriminator loss
        DX_ad_loss = model.discriminator_loss(
            D_X_real_logits, D_X_fake_logits, ls_loss=FLAGS.ls_loss)
        DY_ad_loss = model.discriminator_loss(
            D_Y_real_logits, D_Y_fake_logits, ls_loss=FLAGS.ls_loss)
        discriminator_loss = FLAGS.lambda_gan * (DX_ad_loss + DY_ad_loss)

        # Reconstraction loss
        recon_loss_xyx = model.reconstract_loss(domain_x, xyx_img)
        recon_loss_yxy = model.reconstract_loss(domain_y, yxy_img)
        recon_loss_xx = model.reconstract_loss(domain_x, xx_img)
        recon_loss_yy = model.reconstract_loss(domain_y, yy_img)

        # KL
        kl_xy = model.kl(mu=xy_z)
        kl_yx = model.kl(mu=yx_z)
        kl_xyx = model.kl(mu=xyx_z)
        kl_yxy = model.kl(mu=yxy_z)

        # VAE loss
        vae_loss_x = FLAGS.lambda_rec * recon_loss_xx + \
                     FLAGS.lambda_vae_kl * kl_xy
        vae_loss_y = FLAGS.lambda_rec * recon_loss_yy + \
                     FLAGS.lambda_vae_kl * kl_yx

        # Cycle consistency loss
        cycle_loss_x = FLAGS.lambda_cycle * recon_loss_xyx + \
                       FLAGS.lambda_cycle_kl * kl_xyx # + \
                    #    FLAGS.lambda_cycle_kl * kl_xy
        cycle_loss_y = FLAGS.lambda_cycle * recon_loss_yxy + \
                       FLAGS.lambda_cycle_kl * kl_yxy # + \
                    #    FLAGS.lambda_cycle_kl * kl_yx

        # Optimized generator loss
        opt_generator_loss = generator_loss + \
                             vae_loss_x + vae_loss_y + \
                             cycle_loss_x + cycle_loss_y

        # Optimized discriminator loss
        opt_discrimintor_loss = discriminator_loss

        # Train step
        opt_op = model.train_op(
            opt_generator_loss=opt_generator_loss,
            opt_discrimintor_loss=opt_discrimintor_loss,
            generator_scope='generator',
            discriminator_scope='discriminator',
            generator_lr=FLAGS.generator_lr,
            discriminator_lr=FLAGS.discriminator_lr,
            beta1=FLAGS.beta1)

        summaries = [
            tf.summary.scalar('GX_ad_loss', GX_ad_loss),
            tf.summary.scalar('GY_ad_loss', GY_ad_loss),
            tf.summary.scalar('generator_loss', generator_loss),
            tf.summary.scalar('DX_ad_loss', DX_ad_loss),
            tf.summary.scalar('DY_ad_loss', DY_ad_loss),
            tf.summary.scalar('discriminator_loss', discriminator_loss),
            tf.summary.scalar('recon_loss_xyx', recon_loss_xyx),
            tf.summary.scalar('recon_loss_yxy', recon_loss_yxy),
            tf.summary.scalar('recon_loss_xx', recon_loss_xx),
            tf.summary.scalar('recon_loss_yy', recon_loss_yy),
            tf.summary.scalar('kl_xyx', kl_xyx),
            tf.summary.scalar('kl_yxy', kl_yxy),
            tf.summary.scalar('kl_xy', kl_xy),
            tf.summary.scalar('kl_yx', kl_yx),
            tf.summary.scalar('vae_loss_x', vae_loss_x),
            tf.summary.scalar('vae_loss_y', vae_loss_y),
            tf.summary.scalar('cycle_loss_x', cycle_loss_x),
            tf.summary.scalar('cycle_loss_y', cycle_loss_y),
            tf.summary.scalar('opt_generator_loss', opt_generator_loss),
            tf.summary.scalar('opt_discrimintor_loss', opt_discrimintor_loss)
        ]

        merge_summary = tf.summary.merge(summaries)

        img_buffer = ImageBuffer(buffer_size=50)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start_time = time.time()

        logging.info('Start training...')
        step = 0
        with tqdm(total=FLAGS.iteration) as pbar:
            try:
                while not coord.should_stop():
                    _yx_img, _xy_img = sess.run(
                        [yx_img, xy_img], feed_dict={phase_train: True})

                    x, y = img_buffer(_yx_img, _xy_img, FLAGS.batch_size)

                    outputs = [opt_op,
                               opt_generator_loss,
                               opt_discrimintor_loss,
                               merge_summary]

                    feed_dict = {buffer_x: x, buffer_y: y, phase_train: True}

                    _, _generater_loss, _discriminator_loss, summary_str = \
                        sess.run(outputs, feed_dict=feed_dict)

                    writer.add_summary(summary_str, step)

                    duration = time.time() - start_time
                    training_message = \
                        'step: {} '.format(step + 1) + \
                        'G_loss: {:.3f} '.format(_generater_loss) + \
                        'D_loss: {:.3f} '.format(_discriminator_loss) + \
                        'duration: {:.3f}sec '.format(duration) + \
                        'time_per_step: {:.3f}sec'.format(duration / (step + 1))
                    # print(training_message)
                    pbar.set_description((training_message))
                    pbar.update(FLAGS.batch_size)
                    logging.info(training_message)

                    if not step % FLAGS.snapshot and not step == 0:
                        saver.save(sess, model_dir + '/model', global_step=step)
                        message = 'Saving...\n' + \
                            'Done training for {} steps.'.format(step)
                        logging.info(message)

                    if step == FLAGS.iteration:
                        saver.save(sess, model_dir + '/model', global_step=step)
                        message = 'Saving...\n' + \
                            'Finish training for {} steps.'.format(step)
                        logging.info(message)
                        break

                    step += 1

            except (KeyboardInterrupt, tf.errors.OutOfRangeError):
                coord.request_stop()

            finally:
                coord.request_stop()

            coord.join(threads)


def main(_):
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = os.path.join(
        FLAGS.outdir, FLAGS.arch + '-' + current_time + '-' + str(os.getpid()))
    trained_model = os.path.join(outdir, 'trained_model')
    summary_dir = os.path.join(outdir, 'summary')

    utils.make_dirs(trained_model)
    utils.print_arguments(
        args=FLAGS.__flags, log_fn=os.path.join(outdir, 'args.log'))

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename='{}/train.log'.format(outdir),
        filemode='w', level=logging.INFO)

    train(trained_model, summary_dir)


if __name__ == '__main__':
    tf.app.run()
