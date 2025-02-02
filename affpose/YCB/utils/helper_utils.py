import os
from os import listdir
from os.path import splitext
from glob import glob
import copy

import logging

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

######################
######################

from affpose.YCB import cfg as config

######################
# IMG UTILS
######################

def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def print_depth_info(depth):
    depth = np.array(depth)
    print("Depth of type:{} has min:{} & max:{}".format(depth.dtype, np.min(depth), np.max(depth)))

def print_class_labels(label):
    class_ids = np.unique(np.array(label, dtype=np.uint8))
    class_ids = class_ids[1:] # exclude the backgound
    print("Mask has {} Labels: {}".format(len(class_ids), class_ids))
