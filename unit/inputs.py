import tensorflow as tf


def _preprocessing(x, h, w):
    x = tf.cast(x, tf.float32)
    x = x / 255.
    x = (x - 0.5) / 0.5
    return x


def input_nopair(fn_queue, target_height, target_width, target_channel,
                 batch_size, num_threads, min_after_dequeue,
                 shuffle=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fn_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'x_raw': tf.FixedLenFeature([], tf.string)
        })

    x = tf.decode_raw(features['x_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    x = tf.reshape(x, [height, width, target_channel])
    size = (target_height, target_width)
    x = tf.image.resize_images(x, size)
    
    x = _preprocessing(x, target_height, target_width)
    x = tf.image.resize_image_with_crop_or_pad(
            image=x, target_height=target_height, target_width=target_width)

    capacity = min_after_dequeue + num_threads * batch_size
    if shuffle:
        _x = tf.train.shuffle_batch(
            [x],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads,
            min_after_dequeue=min_after_dequeue)
    else:
        _x = tf.train.batch(
            [x],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads)

    return _x


def input_pair(fn_queue, target_height, target_width, target_channel, 
               batch_size, num_threads, min_after_dequeue,
               shuffle=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fn_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'x_raw': tf.FixedLenFeature([], tf.string),
            'y_raw': tf.FixedLenFeature([], tf.string)
        })

    x = tf.decode_raw(features['x_raw'], tf.uint8)
    y = tf.decode_raw(features['y_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    x = tf.reshape(x, [height, width, 3])
    y = tf.reshape(y, [height, width, target_channel])

    size = (target_height, target_width)
    x = tf.image.resize_images(x, size)
    y = tf.image.resize_images(y, size)
    x = _preprocessing(x, target_height, target_width)
    y = _preprocessing(y, target_height, target_width)

    x = tf.image.resize_image_with_crop_or_pad(
            image=x, target_height=target_height, target_width=target_width)
    y = tf.image.resize_image_with_crop_or_pad(
            image=y, target_height=target_height, target_width=target_width)
    capacity = min_after_dequeue + num_threads * batch_size

    if shuffle:
        _x, _y = tf.train.shuffle_batch(
            [x, y],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads,
            min_after_dequeue=min_after_dequeue)
    else:
        _x, _y = tf.train.batch(
            [x, y],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads)

    return _x, _y