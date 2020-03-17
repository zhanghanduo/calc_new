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

    sample['img'] = tf.reshape(tf.cast(tf.decode_raw(sample['img'], tf.uint8),
                                   tf.float32) / 255.0, [__vh, __vw, 3])
    sample['label'] = tf.reshape(tf.decode_raw(sample['label'], tf.uint8), [__vh, __vw])
    sample['label'] = tf.cast(tf.one_hot(sample['label'], N_CLASSES), tf.float32)

    # image = tf.image.decode_image(sample['img'])
    # print("image dim: ", tf.shape(sample['img']))
    # img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
    # label = sample['label']
    # filename = sample['filename']
    return sample


class TFRecordExtractor:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)

    def extract_image(self):
        # Create folder to store extracted images
        folder_path = './ExtractedImages'
        shutil.rmtree(folder_path, ignore_errors=True)
        os.mkdir(folder_path)

        # Pipeline of dataset and iterator
        # print("path: ", self.tfrecord_file)
        files = tf.data.Dataset.list_files(self.tfrecord_file)
        dataset = tf.data.TFRecordDataset(files)
        # dataset.map(_extract_fn, num_parallel_calls=n_cpus() // 2).batch(12)
        # dataset = dataset.prefetch(buffer_size=2)
        dataset = dataset.map(_extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            i = 0
            # try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                image_data = sess.run(next_image_data)
                # img_ = Image.fromarray(image_data['img'], 'RGB')
                # img_.show()
                # print(image_data)

                image_name1 = 'raw' + str(i)
                image_name2 = 'aug' + str(i)
                i = i+1
                save_path1 = os.path.abspath(os.path.join(folder_path, image_name1))
                save_path2 = os.path.abspath(os.path.join(folder_path, image_name2))

                im_l = tf.concat([image_data['img'], image_data['label']], axis=-1)
                # # x = tf.image.random_flip_left_right(im_l)
                x = tf.image.random_crop(im_l, [vh, vw, 3 + N_CLASSES])
                images = x[np.newaxis, :, :, :3]
                # labels = x[:, :, :, 3:]

                im_warp = tf.image.random_flip_left_right(images)
                im_warp = layers.rand_warp(im_warp, [vh, vw])
                im_w_adj = tf.clip_by_value(im_warp + \
                                            tf.random.uniform([tf.shape(im_warp)[0], 1, 1, 1], -.8, 0.0),
                                            0.0, 1.0)
                tf.where(tf.less(tf.reduce_mean(im_warp, axis=[1, 2, 3]), 0.2), im_warp, im_w_adj)
                # im_warp_v = tf.Variable(im_warp)
                im_warp_v = layers.random_erasing(im_warp)
                im_warp_v = tf.squeeze(layers.random_erasing(im_warp_v))
                # print(type(im_warp_v.eval()))

                mpimg.imsave(save_path1, image_data['img'])
                # print(im_warp_v.dtype)
                # img_tosave = tf.squeeze(im_warp)
                mpimg.imsave(save_path2, im_warp_v.eval())
                # print('Save path = ', save_path1, ', Label = ', image_data['label'])
            # except:
            #     print("wrong")


if __name__ == '__main__':
    indir = FLAGS.input_dir
    infile = 'train_data0.tfrecord'
    t = TFRecordExtractor(indir + infile)
    t.extract_image()
