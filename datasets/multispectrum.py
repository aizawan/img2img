import os
import sys
import argparse
import glob
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from distutils.util import strtobool

sys.path.insert(0, os.getcwd())
from datasets import tfrecord
from utils import utils


def make_pairs_on_multispectrum(root, mode):
    if mode == 'train':
        sets = ('set00', 'set01', 'set02', 'set03', 'set04', 'set05')
    elif mode == 'test':
        sets = ('set06', 'set07', 'set08', 'set09', 'set10', 'set11')
    elif mode == 'train-day':
        sets = ('set00', 'set01', 'set02')
    elif mode == 'train-night':
        sets = ('set03', 'set04', 'set05')
    elif mode == 'test-day':
        sets = ('set06', 'set07', 'set08')
    elif mode == 'test-night':
        sets = ('set09', 'set10', 'set11')
    else:
        print('error', mode)

    pairs = []
    for setn in sets:
        search_path = root + "/{}/*/visible/*".format(setn)
        for fn in sorted(glob.glob(search_path)):
            pairs.append((fn, fn.replace('visible', 'lwir')))

    return pairs


def _read_img_ids(img_dir):
    city = os.listdir(img_dir)
    img_ids = []
    for _city in city:
        search_dir = os.path.join(img_dir, _city)
        img_ids.extend(
            [_city + "/" + img_path.split("_leftImg8bit.png")[0]
                for img_path in sorted(os.listdir(search_dir))])
    return img_ids


def _convert_pair_to_tfrecord(pair, writer):
    for x_path, y_path in pair:
        x = np.array(Image.open(x_path))
        y = np.array(Image.open(y_path))
        h = x.shape[0]
        w = x.shape[1]

        x_raw = x.tostring()
        y_raw = y.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tfrecord.int64_feature(h),
            'width': tfrecord.int64_feature(w),
            'x_raw': tfrecord.bytes_feature(x_raw),
            'y_raw': tfrecord.bytes_feature(y_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def _convert_nopair_to_tfrecord(x_sets, writer):
    for x_path in x_sets:
        x = np.array(Image.open(x_path))
        h = x.shape[0]
        w = x.shape[1]

        x_raw = x.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tfrecord.int64_feature(h),
            'width': tfrecord.int64_feature(w),
            'x_raw': tfrecord.bytes_feature(x_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def convert_to_tfrecord(x, outdir, fname):
    utils.make_dirs(outdir)
    
    fn = os.path.join(outdir, fname + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(fn)
    print('Writing', fn)
    if isinstance(x[0], (list, tuple)):
        _convert_pair_to_tfrecord(x, writer)
    else:
        _convert_nopair_to_tfrecord(x, writer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--path_list', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--pair', type=strtobool, default=True)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    
    print('-*-' * 5, args.dataset, '-*-' * 5)
    if args.dataset == 'multispectrum':
        pairs = make_pairs_on_multispectrum(args.root, args.mode)
        x = pairs
    elif args.dataset == 'multispectrum-vis':
        pairs = make_pairs_on_multispectrum(args.root, args.mode)
        x = [pair[0] for pair in pairs]
    elif args.dataset == 'multispectrum-lwir':
        pairs = make_pairs_on_multispectrum(args.root, args.mode)
        x = [pair[1] for pair in pairs]

    if args.num_samples is not None:
        step = len(x) // args.num_samples
        x = [x[i * step] for i in range(args.num_samples)]

    if not args.pair:
        random.shuffle(x)

    fname = args.dataset + '-' + args.mode + '-' + str(len(x))
    convert_to_tfrecord(x, args.outdir, fname)


if __name__ == '__main__':
    main()