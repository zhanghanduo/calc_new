#!/usr/bin/env python3

import layers
import cv2
from os.path import join
import numpy as np
# import tensorflow as tf
import Augmentor

vw = 320
vh = 320


class Augment:
    def __init__(self):
        self.w = 2 * 640
        self.h = 2 * 480
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def update(self, _im):
        # sc = .4
        # h, w = (int(sc * _im.shape[0]), int(sc * _im.shape[1]))
        vh = _im.shape[0]
        vw = _im.shape[1]
        # im = cv2.resize(_im, (w, h))
        self.canvas[100:(100 + vh), :vw, :] = _im

        cv2.putText(self.canvas, "Original", (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
        cv2.putText(self.canvas, "Distorted", (0, 150 + vh), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))

        self.canvas[(200 + vh):(200 + 2 * vh), :vw, :] = _im

        cv2.imshow("Image Augment", self.canvas)
        cv2.waitKey(0)


if __name__ == "__main__":
    data_root = "/mnt/4102422c-af52-4b55-988f-df7544b35598/dataset/KITTI/KITTI_Odometry/"
    seq = "14"
    vo_fn = data_root + "dataset/poses/" + seq.zfill(2) + ".txt"
    im_dir = data_root + "dataset/sequences/" + seq.zfill(2)

    aux_dir = "/home/handuo/projects/paper/image_base/downloads"

    i = 0
    gui = Augment()

    # with tf.Session() as sess:
    ims = []
    p = Augmentor.Pipeline(join(im_dir, "image_0/"), output_directory="../../../output", save_format="JPEG")
    # print("Has %s samples." % (len(p.augmentor_images)))
    p.zoom(probability=0.3, min_factor=0.9, max_factor=1.2)
    p.skew(probability=0.75, magnitude=0.3)
    # p.random_erasing(probability=0.5, rectangle_area=0.3)
    p.multi_erasing(probability=0.5, max_x_axis=0.3, max_y_axis=0.15, max_num=4)
    # p.rotate(probability=0.5, max_left_rotation=6, max_right_rotation=6)
    p.sample(10)


