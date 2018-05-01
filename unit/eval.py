""" eval.py
    The code to evaluate trained model for image-to-image translation.
"""

import os
import sys
import time
import logging
import importlib
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.getcwd())
from unit import inputs
from utils import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('use_gpu_server', True, 'use_gpu_server')
flags.DEFINE_string('arch', 'model', 'Network architecure')
flags.DEFINE_string('outdir', 'output/unit', 'Output directory')
flags.DEFINE_string('checkpoint_dir', 'output/unit/model/trained_model',
    'Directory where to read model checkpoint.')
flags.DEFINE_string('x_tfrecord',
    '/tmp/data/unit/x-train-1000.tfrecord', 'TFRecord path')
flags.DEFINE_string('y_tfrecord',
    '/tmp/data/unit/y-train-1000.tfrecord', 'TFRecord path')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('target_height', 256, 'Input height')
flags.DEFINE_integer('target_width', 256, 'Input width')
flags.DEFINE_integer('target_channel', 3, 'Input channel')
flags.DEFINE_integer('num_enc_res_block', 3, 'residual block')
flags.DEFINE_integer('num_shared_res_block', 2, 'residual block')
flags.DEFINE_integer('num_dec_res_block', 3, 'residual block')
flags.DEFINE_integer('num_threads', 8, 'Number of threads to read batches')
flags.DEFINE_integer('min_after_dequeue', 10, 'min_after_dequeue')
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_integer('num_sample', 10, 'Number of sample to eval the model.')

if FLAGS.use_gpu_server:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


def evaluate(res_dir):
    logging.info('Evaluate {}'.format(FLAGS.arch))
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

        plots = {
            "X": domain_x,
            "Y": domain_y,
            "X-Y": xy_img,
            "Y-X": yx_img,
            "X-X": xx_img,
            "Y-Y": yy_img,
            "X-Y-X": xyx_img,
            "Y-X-Y": yxy_img
        }

        plots_list = list(plots.values())
        plots_key = list(plots.keys())
        print(plots_key)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(graph=graph) as sess:
        sess.run(init_op)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            meta_graph_path = ckpt.model_checkpoint_path + ".meta"
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = os.path.splitext(os.path.basename(meta_graph_path))[0]
            res_dir = os.path.join(res_dir, step)
            utils.make_dirs(res_dir)
        else:
            print('No checkpoint file found')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start_time = time.time()

        logging.info('Start evaluating...')

        step = 1
        try:
            while not coord.should_stop():
                outputs = sess.run(plots_list, feed_dict={phase_train: False})

                for v, k in zip(outputs, plots_key):
                    _res_dir = os.path.join(res_dir, k)
                    _res_path = os.path.join(_res_dir, "{}.png".format(step))
                    utils.make_dirs(_res_dir)

                    v = np.squeeze(np.uint8((v * 0.5 + 0.5) * 255))
                    img = Image.fromarray(v)
                    img.save(_res_path)

                    logging.info("Save {}".format(_res_path))
                    print("Save {}".format(_res_path))

                step += 1
                if FLAGS.num_sample < step: break

        except tf.errors.OutOfRangeError:
            logging.info('Finished.')

        except KeyboardInterrupt:
            coord.request_stop()

        finally:
            coord.request_stop()

        coord.join(threads)


def main(_):
    utils.make_dirs(FLAGS.outdir)

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename='{}/eval.log'.format(FLAGS.outdir),
        filemode='w', level=logging.INFO)

    evaluate(FLAGS.outdir)


if __name__ == '__main__':
    tf.app.run()
