#!/usr/bin/env python3

import os
import sys
import tensorflow as tf
import numpy as np
from time import time

import utils
import layers
from dataset.coco_classes import calc_classes

import shutil
import matplotlib.image as mpimg

N_CLASSES = len(calc_classes.keys())

from multiprocessing import cpu_count as n_cpus

from dataset.gen_tfrecords import vw as __vw
from dataset.gen_tfrecords import vh as __vh

vw = 256
vh = 192  # Need 128 since we go down by factors of

FLAGS = tf.app.flags.FLAGS
if __name__ == '__main__':
    tf.app.flags.DEFINE_string("mode", "train", "train, pr, ex, or best")

    tf.app.flags.DEFINE_string("model_dir", "model", "Estimator model_dir")
    tf.app.flags.DEFINE_string("data_dir", "dataset/CampusLoopDataset", "Path to data")
    tf.app.flags.DEFINE_string("title", "Precision-Recall Curve", "Plot title")
    tf.app.flags.DEFINE_integer("n_include", 5, "")

    tf.app.flags.DEFINE_integer("steps", 200000, "Training steps")
    tf.app.flags.DEFINE_string(
        "hparams", "",
        "A comma-separated list of `name=value` hyperparameter values. This flag "
        "is used to override hyperparameter settings when manually "
        "selecting hyperparameters.")

    tf.app.flags.DEFINE_integer("batch_size", 12, "Size of mini-batch.")

    tf.app.flags.DEFINE_string("input_dir", "/mnt/4102422c-af52-4b55-988f-df7544b35598/dataset/coco/calc_tfrecords/",
                               "tfrecords dir")
    tf.app.flags.DEFINE_boolean("include_calc", False, "Include original calc in pr curve"
                                                       "Place in 'calc_model' directory if this is set")
    tf.app.flags.DEFINE_string("image_fl", "dataset/examples/bon.png", "Example image location")


def _extract_fn(tfrecord):
    # Extract features using the keys set during creation
    # features = {
    #     'filename': tf.FixedLenFeature([], tf.string),
    #     'rows': tf.FixedLenFeature([], tf.int64),
    #     'cols': tf.FixedLenFeature([], tf.int64),
    #     'channels': tf.FixedLenFeature([], tf.int64),
    #     'image': tf.FixedLenFeature([], tf.string),
    #     'label': tf.FixedLenFeature([], tf.int64)
    # }
    features = {'img': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
                # 'cl_live': tf.FixedLenFeature([], tf.string),
                # 'cl_mem': tf.FixedLenFeature([], tf.string)
                }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    image = tf.image.decode_image(sample['img'])
    # img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
    label = sample['label']
    # filename = sample['filename']
    return [image, label]


class TFRecordExtractor:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)

    def extract_image(self):
        # Create folder to store extracted images
        folder_path = './ExtractedImages'
        shutil.rmtree(folder_path, ignore_errors=True)
        os.mkdir(folder_path)

        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(_extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            try:
                # Keep extracting data till TFRecord is exhausted
                while True:
                    image_data = sess.run(next_image_data)

                    # Check if image shape is same after decoding
                    if not np.array_equal(image_data[0].shape, image_data[3]):
                        print('Image {} not decoded properly'.format(image_data[2]))
                        continue

                    save_path = os.path.abspath(os.path.join(folder_path, image_data[2].decode('utf-8')))
                    mpimg.imsave(save_path, image_data[0])
                    print('Save path = ', save_path, ', Label = ', image_data[1])
            except:
                pass


def create_input_fn(split, batch_size):
    """Returns input_fn for tf.estimator.Estimator.

    Reads tfrecord file and constructs input_fn for training

    Args:
    tfrecord: the .tfrecord file
    batch_size: The batch size!

    Returns:
    input_fn for tf.estimator.Estimator.

    Raises:
    IOError: If test.txt or dev.txt are not found.
    """

    def input_fn():
        """input_fn for tf.estimator.Estimator."""

        indir = FLAGS.input_dir
        tfrecord = 'train_data*.tfrecord' if split == 'train' else 'validation_data.tfrecord'

        def parser(serialized_example):

            features_ = {'img': tf.FixedLenFeature([], tf.string),
                         'label': tf.FixedLenFeature([], tf.string)}

            if split != 'train':
                features_['cl_live'] = tf.FixedLenFeature([], tf.string)
                features_['cl_mem'] = tf.FixedLenFeature([], tf.string)

            fs = tf.parse_single_example(
                serialized_example,
                features=features_
            )

            fs['img'] = tf.reshape(tf.cast(tf.decode_raw(fs['img'], tf.uint8),
                                           tf.float32) / 255.0, [__vh, __vw, 3])
            fs['label'] = tf.reshape(tf.decode_raw(fs['label'], tf.uint8), [__vh, __vw])
            fs['label'] = tf.cast(tf.one_hot(fs['label'], N_CLASSES), tf.float32)
            if split != 'train':
                fs['cl_live'] = tf.reshape(tf.cast(tf.decode_raw(fs['cl_live'], tf.uint8),
                                                   tf.float32) / 255.0, [__vh, __vw, 3])
                fs['cl_mem'] = tf.reshape(tf.cast(tf.decode_raw(fs['cl_mem'], tf.uint8),
                                                  tf.float32) / 255.0, [__vh, __vw, 3])
                fs['cl_live'] = tf.reshape(tf.image.resize_images(fs['cl_live'],
                                                                  (vh, vw)), [vh, vw, 3])
                fs['cl_mem'] = tf.reshape(tf.image.resize_images(fs['cl_mem'],
                                                                 (vh, vw)), [vh, vw, 3])

            return fs

        if split == 'train':
            files = tf.data.Dataset.list_files(indir + tfrecord, shuffle=True,
                                               seed=np.int64(time()))
        else:
            files = [indir + tfrecord]

        dataset = tf.data.TFRecordDataset(files)
        # dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(400, seed=np.int64(time())))
        dataset.shuffle(
            400, seed=np.int64(time()), reshuffle_each_iteration=True).repeat()
        dataset.map(parser, num_parallel_calls=n_cpus() // 2).batch(batch_size if split == 'train' else batch_size // 3)

        dataset = dataset.prefetch(buffer_size=2)

        return dataset

    return input_fn


if __name__ == '__main__':
    indir = FLAGS.input_dir
    infile = 'train_data0.tfrecord'
    t = TFRecordExtractor(indir + infile)
    t.extract_image()
