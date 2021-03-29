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

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).resolve().parents[1]

import cfg as config

######################
# IMG UTILS
######################

def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def print_depth_info(depth):
    depth = np.array(depth)
    print(f"Depth of type:{depth.dtype} has min:{np.min(depth)} & max:{np.max(depth)}")

def print_class_labels(label):
    class_ids = np.unique(np.array(label, dtype=np.uint8))
    class_ids = class_ids[1:] # exclude the backgound
    print(f"Mask has {len(class_ids)} Labels: {class_ids}")
