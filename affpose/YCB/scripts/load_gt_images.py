import glob
import numpy as np

import cv2
from PIL import Image

#######################################
#######################################

from affpose.utils import helper_utils

from affpose.YCB import cfg as config
from affpose.YCB.utils.dataset import ycb_dataset_utils

#######################################
#######################################

def main():

    # image_files = open('{}'.format(config.TRAIN_FILE), "r")
    image_files = open('{}'.format(config.TEST_FILE), "r")
    image_files = image_files.readlines()
    print("Loaded Files: {}".format(len(image_files)))

    # select random test images
    np.random.seed(0)
    num_files = 100
    random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    image_files = np.array(image_files)[random_idx]
    print("Chosen Files: {}".format(len(image_files)))

    for image_idx, image_addr in enumerate(image_files):

        image_addr = image_addr.rstrip()

        rgb_addr   = config.DATASET_ROOT_PATH + image_addr + config.RGB_EXT
        depth_addr = config.DATASET_ROOT_PATH + image_addr + config.DEPTH_EXT
        label_addr    = config.DATASET_ROOT_PATH + image_addr + config.LABEL_EXT

        rgb      = np.array(Image.open(rgb_addr))
        depth    = np.array(Image.open(depth_addr))
        label    = np.array(Image.open(label_addr))

        #####################
        # DEPTH INFO
        #####################

        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        #####################
        # LABEL INFO
        #####################

        helper_utils.print_class_labels(label)

        #####################
        # PLOTTING
        #####################

        rgb = cv2.resize(rgb, config.RESIZE)
        depth = cv2.resize(depth, config.RESIZE)
        label = cv2.resize(label, config.RESIZE)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('label', ycb_dataset_utils.colorize_obj_mask(label))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()