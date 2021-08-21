import unittest

import numpy as np
import numpy.ma as ma

import cv2
from PIL import Image

import scipy.io as scio

#######################################
#######################################

import sys
sys.path.append('../../..')

#######################################
#######################################

from affpose.YCB_Aff.dataset import dataloader
from affpose.YCB_Aff.dataset import ycb_aff_dataset_utils

#######################################
#######################################


class TestARLAffPoseDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestARLAffPoseDataloader, self).__init__(*args, **kwargs)
        # load real images.
        self.dataloader = dataloader.YCBAff(split='test', select_random_images=True, num_images=10)

    def load_images(self):

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.get_item(image_idx)

            #####################
            #####################

            rgb = data["rgb"]
            obj_label = data["obj_label"]
            aff_label = data["aff_label"]
            depth_8bit = data["depth_8bit"]

            #####################
            #####################

            colour_obj_label = ycb_aff_dataset_utils.colorize_obj_mask(obj_label)
            colour_obj_label = cv2.addWeighted(rgb, 0.35, colour_obj_label, 0.65, 0)

            colour_aff_label = ycb_aff_dataset_utils.colorize_aff_mask(aff_label)
            colour_aff_label = cv2.addWeighted(rgb, 0.35, colour_aff_label, 0.65, 0)

            #####################
            # PLOTTING
            #####################

            cv2.imshow('colour_obj_label', cv2.cvtColor(colour_obj_label, cv2.COLOR_BGR2RGB))
            cv2.imshow('colour_obj_part_label', cv2.cvtColor(colour_aff_label, cv2.COLOR_BGR2RGB))
            cv2.imshow('depth', depth_8bit)

            cv2.waitKey(0)

    def load_gt_pose(self):

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.draw_gt_obj_pose(image_idx, project_mesh_on_image=False, verbose=True)

            #####################
            #####################

            depth_8bit = data["depth_8bit"]
            cv2_obj_pose_img = data["cv2_obj_pose_img"]
            cv2_obj_part_pose_img = data["cv2_obj_part_pose_img"]

            #####################
            # PLOTTING
            #####################

            cv2.imshow('depth', depth_8bit)
            cv2.imshow('cv2_obj_pose_img', cv2.cvtColor(cv2_obj_pose_img, cv2.COLOR_BGR2RGB))
            cv2.imshow('cv2_obj_part_pose_img', cv2.cvtColor(cv2_obj_part_pose_img, cv2.COLOR_BGR2RGB))

            cv2.waitKey(0)

if __name__ == '__main__':
    # run all test.
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(TestARLAffPoseDataloader("load_images"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
