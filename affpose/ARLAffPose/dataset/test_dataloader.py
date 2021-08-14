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

from affpose.ARLAffPose.dataset import dataloader

#######################################
#######################################


class TestARLAffPoseDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestARLAffPoseDataloader, self).__init__(*args, **kwargs)
        # load real images.
        self.dataloader = dataloader.ARLAffPose(split='test',
                                                select_random_images=True,
                                                num_images=250)

    def load_images(self):

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.get_item(image_idx)

            #####################
            #####################

            rgb = data["rgb"]
            depth_8bit = data["depth_8bit"]

            #####################
            # PLOTTING
            #####################

            cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            cv2.imshow('depth', depth_8bit)

            cv2.waitKey(0)

    def load_gt_pose(self):

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.draw_gt_obj_pose(image_idx, project_mesh_on_image=False)

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
    suite.addTest(TestARLAffPoseDataloader("load_gt_pose"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
