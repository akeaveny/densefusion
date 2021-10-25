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

from affpose.ARLAffPose import cfg as config
from affpose.ARLAffPose.dataset import dataloader
from affpose.ARLAffPose.dataset import arl_affpose_dataset_utils
from affpose.ARLAffPose.utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

SPLIT = 'train'
NUM_PT_MIN = 500


class TestARLAffPoseDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestARLAffPoseDataloader, self).__init__(*args, **kwargs)
        # load real images.
        self.dataloader = dataloader.ARLAffPose(split=SPLIT)

    def check_masked_depth(self):

        bad_masked_depth_images = []

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.get_item(image_idx)

            #####################
            #####################

            rgb = data["rgb"]
            depth_16bit = data["depth_16bit"]
            depth_8bit = data["depth_8bit"]
            obj_label = data["obj_label"]
            obj_part_label = data["obj_part_label"]
            cv2_obj_pose_img = data["cv2_obj_pose_img"]
            cv2_obj_part_pose_img = data["cv2_obj_part_pose_img"]
            meta = data["meta"]

            #######################################
            # OBJECT
            #######################################

            obj_ids = np.array(meta['object_class_ids']).flatten()
            label_obj_ids = np.unique(obj_label)[1:]
            label_obj_part_ids = np.unique(obj_part_label)[1:]

            for idx, obj_id in enumerate(obj_ids):
                if obj_id in label_obj_ids:
                    obj_id = int(obj_id)
                    obj_name = "{:<15}".format(arl_affpose_dataset_utils.map_obj_id_to_name(obj_id))

                    # get obj mask.
                    masked_obj_label = np.ma.getmaskarray(np.ma.masked_equal(obj_label, obj_id))
                    mask_obj_depth = masked_obj_label * ma.getmaskarray(ma.masked_not_equal(depth_16bit, 0))
                    num_obj_points = len(mask_obj_depth.nonzero()[0])

                    #######################################
                    # ITERATE OVER OBJ PARTS
                    #######################################

                    obj_part_ids = arl_affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
                    for obj_part_id in obj_part_ids:
                        if obj_part_id in label_obj_part_ids and obj_part_id in self.dataloader.obj_part_ids:
                            obj_part_id = int(obj_part_id)
                            aff_id = arl_affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                            aff_name = "{:<15}".format(arl_affpose_dataset_utils.map_aff_id_to_name(aff_id))

                            # get obj part mask.
                            masked_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(obj_part_label, obj_part_id))
                            mask_obj_part_depth = masked_obj_part_label * ma.getmaskarray(ma.masked_not_equal(depth_16bit, 0))
                            num_obj_part_points = len(mask_obj_part_depth.nonzero()[0])
                            if num_obj_part_points < NUM_PT_MIN:
                                print('\nimage:{}/{}, file:{}'.format(image_idx + 1, len(self.dataloader.img_files), image_addr.rstrip()))
                                print("\tObject: {} Masked Depth: {}".format(obj_name, num_obj_points))
                                print("\t\tAff: {}\tMasked Depth: {}".format(aff_name, num_obj_part_points))
                                bad_masked_depth_images.append(image_addr)

                                # #######################################
                                # # BBOX
                                # #######################################
                                #
                                # obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_bbox(obj_part_label.copy(), obj_part_id, config.HEIGHT, config.WIDTH,config.BORDER_LIST)
                                # depth_8bit = cv2.rectangle(depth_8bit, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), 128, 2)
                                #
                                # #####################
                                # # PLOTTING
                                # #####################
                                #
                                # cv2.imshow('heatmap', cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET))
                                # cv2.waitKey(1)


        bad_masked_depth_images = np.unique(np.array(bad_masked_depth_images))

        if SPLIT == "train":
            file = open(config.FORMATTED_TRAIN_FILE, 'w')
        elif SPLIT == "val":
            file = open(config.FORMATTED_VAL_FILE, 'w')
        elif SPLIT == "test":
            file = open(config.FORMATTED_TEST_FILE, 'w')

        count = 0
        for idx, image in enumerate(self.dataloader.img_files):
            if image in bad_masked_depth_images:
                continue
            file.write(image.rstrip())
            file.write('\n')
            count += 1
        print('wrote {} files'.format(count))

if __name__ == '__main__':
    # run all test.
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(TestARLAffPoseDataloader("check_masked_depth"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
