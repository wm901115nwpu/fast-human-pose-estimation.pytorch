# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Roman Tsyganok (iskullbreakeri@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import division
import argparse

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image

import gluoncv as gcv
from gluoncv import data
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import plot_keypoints
from matplotlib import pyplot as plt

import cv2
import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Show demo')

    parser.add_argument('--img',
                        help='picture filename',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()

    parser.add_argument('--model',
                        help='model state file',
                        required=True,
                        type=str)
    parser.add_argument('--type',
                        help='ONNX/OpenVINO',
                        required=False,
                        type=str)

    parser.add_argument('--xml',
                        help='OpenVINO XML config',
                        required=False,
                        type=str)
    parser.add_argument('--backend',
                        help='OpenCV DNN Backend',
                        required=False,
                        type=str)
    parser.add_argument('--width',
                        help='input network width',
                        required=True,
                        type=int)
    parser.add_argument('--height',
                        help='input network height',
                        required=True,
                        type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    network = None

    scale = 1.0

    detector = get_model('yolo_darknet53_coco', pretrained=True)
    detector.reset_class(['person'], reuse_weights=['person'])

    if args.type == 'ONNX':
        network = cv2.dnn.readNetFromONNX(args.model)

    elif args.type == 'OpenVINO':
        network = cv2.dnn.readNetFromModelOptimizer(args.xml, args.model)

    # default backend if wasn`t specified
    if not args.backend:
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

    # in case you are going to use CUDA backend in OpenCV, make sure that opencv built with CUDA support
    elif args.backend == 'CUDA':
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # in case you are going to use OpenVINO model, make sure that inference engine already installed and opencv built with IE support
    elif args.backend == 'INFERENCE':
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    x, img = data.transforms.presets.yolo.load_test(args.img, short=512)
    class_IDs, scores, bounding_boxes = detector(x)

    pose_input, upscaled_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxes)

    pose_input = pose_input.asnumpy()
    bs = []
    for i in range(pose_input.shape[0]):
        input = cv2.dnn.blobFromImage(np.transpose(np.squeeze(pose_input[i,:,:,:]),(1,2,0)), scale, (args.width, args.height), (0, 0, 0), False);
        network.setInput(input)
        temp = network.forward()
        bs.append(temp)

    output = np.concatenate(bs, axis=0)

    output = mx.nd.array(output)
    pred_coords, confidence = heatmap_to_coord(output, upscaled_bbox)

    ax = plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxes, scores,
                        box_thresh=0.5, keypoint_thresh=0.2)
    plt.show()

if __name__ == '__main__':
    main()