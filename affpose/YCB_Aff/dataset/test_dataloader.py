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

from affpose.YCB_Aff import cfg as config
from affpose.YCB_Aff.dataset import dataloader
from affpose.YCB_Aff.dataset import ycb_aff_dataset_utils

#######################################
#######################################


class TestARLAffPoseDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestARLAffPoseDataloader, self).__init__(*args, **kwargs)
        # load real images.
        self.dataloader = dataloader.YCBAff(split='train',
                                            select_random_images=True,
                                            num_images=100)

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

    def get_occlusion_metrics(self):

        total_obj_occlusion = np.zeros(shape=(len(self.dataloader.img_files), config.NUM_OBJECTS))
        total_obj_part_occlusion = np.zeros(shape=(len(self.dataloader.img_files), config.NUM_OBJECTS))

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.draw_gt_obj_pose(image_idx, get_occlusion_metrics=True)

            #####################
            #####################

            obj_part_label = data["obj_part_label"]
            obj_part_occlusion_mask = data["obj_part_occlusion_mask"]

            obj_occlusion_dict = {obj_id: 0 for obj_id in range(1, config.NUM_OBJECTS + 1)}
            obj_part_occlusion_dict = {obj_part_id_1: 0 for obj_part_id_1 in range(1, config.NUM_OBJECTS_PARTS + 1)}

            ####################
            # occlusion from overlap
            ####################

            obj_part_ids = list(obj_part_occlusion_mask.keys())
            for obj_part_id_1 in obj_part_ids:
                if obj_part_id_1 in ycb_aff_dataset_utils.DRAW_OBJ_PART_POSE:
                    obj_id_1 = ycb_aff_dataset_utils.map_obj_part_ids_to_obj_id(obj_part_id_1)
                    obj_part_mask_1 = obj_part_occlusion_mask[obj_part_id_1]

                    for obj_part_id_2 in obj_part_ids:
                        if obj_part_id_1 != obj_part_id_2:
                            obj_id_2 = ycb_aff_dataset_utils.map_obj_part_ids_to_obj_id(obj_part_id_2)
                            obj_part_mask_2 = obj_part_occlusion_mask[obj_part_id_2]

                            #####################
                            # intersection
                            #####################

                            intersection = cv2.bitwise_and(obj_part_mask_1, obj_part_mask_2).astype(np.uint8)
                            intersection_points = np.count_nonzero(intersection)

                            if intersection_points:

                                # remove duplicate.
                                idx = int(np.where(np.array(obj_part_ids) == obj_part_id_2)[0])
                                obj_part_ids.pop(idx)

                                # get obj_part_id.
                                idxs = np.nonzero(intersection)
                                rows, cols = idxs[0], idxs[1]
                                mask_obj_part_ids = obj_part_label[rows, cols]
                                counts = np.bincount(mask_obj_part_ids)
                                mask_obj_part_id = np.argmax(counts)

                                # get occlusion
                                if mask_obj_part_id == obj_part_id_2:
                                    occlusion = intersection_points / np.count_nonzero(obj_part_mask_1)
                                    obj_part_occlusion_dict[obj_part_id_1] += occlusion
                                    # add to object masks.
                                    # expected_obj_mask[obj_id_1] += np.count_nonzero(obj_part_mask_1)
                                    # actual_obj_mask[obj_id_1] += intersection_points
                                    obj_occlusion_dict[obj_id_1] += occlusion

                                elif mask_obj_part_id == obj_part_id_1:
                                    occlusion = intersection_points / np.count_nonzero(obj_part_mask_2)
                                    obj_part_occlusion_dict[obj_part_id_2] += occlusion
                                    # add to object masks.
                                    # expected_obj_mask[obj_id_2] += np.count_nonzero(obj_part_mask_2)
                                    obj_occlusion_dict[obj_id_2] += occlusion

            # for obj_id_1, expected_points in expected_obj_mask.items():
            #     intersection_points = actual_obj_mask[obj_id_1]
            #     if intersection_points > 0:
            #         occlusion = intersection_points / expected_points
            #         obj_occlusion_dict[obj_id_1] += occlusion

            #####################
            #####################

            for obj_id, obj_occlusion in obj_occlusion_dict.items():
                if obj_occlusion > 0:
                    obj_name = "{}".format(ycb_aff_dataset_utils.map_obj_id_to_name(obj_id))
                    total_obj_occlusion[image_idx, obj_id - 1] = obj_occlusion
                    print(f'{obj_name}: {obj_occlusion * 100:.2f} [%]')
                    obj_part_ids = ycb_aff_dataset_utils.map_obj_ids_to_obj_part_ids(obj_id)
                    for obj_part_id in obj_part_ids:
                        if obj_part_id in ycb_aff_dataset_utils.DRAW_OBJ_PART_POSE:
                            aff_id = ycb_aff_dataset_utils.map_obj_part_ids_to_aff_ids(obj_part_id)
                            aff_name = "{}".format(ycb_aff_dataset_utils.map_aff_id_to_name(aff_id))
                            obj_part_occlusion = obj_part_occlusion_dict[obj_part_id]
                            if obj_part_occlusion > 0:
                                total_obj_part_occlusion[image_idx, obj_id - 1] = obj_part_occlusion
                                print(f'\t{aff_name}: {obj_part_occlusion * 100:.2f} [%]')

            cv2.imshow('rgb', cv2.cvtColor(data["rgb"], cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)

        print()
        for obj_id in range(config.NUM_OBJECTS):
            obj_name = "{:<25}".format(ycb_aff_dataset_utils.map_obj_id_to_name(obj_id + 1))
            mean_obj_occlusion = np.mean(total_obj_occlusion[:, obj_id])
            max_obj_occlusion = np.max(total_obj_occlusion[:, obj_id])
            mean_obj_part_occlusion = np.mean(total_obj_part_occlusion[:, obj_id])
            max_obj_part_occlusion = np.max(total_obj_part_occlusion[:, obj_id])
            obj_part_ids = ycb_aff_dataset_utils.map_obj_ids_to_obj_part_ids(obj_id + 1)
            for obj_part_id in obj_part_ids:
                if obj_part_id in ycb_aff_dataset_utils.DRAW_OBJ_PART_POSE:
                    aff_id = ycb_aff_dataset_utils.map_obj_part_ids_to_aff_ids(obj_part_id)
                    aff_name = "{:<15}".format(ycb_aff_dataset_utils.map_aff_id_to_name(aff_id))
            print(f'{obj_name}',
                  f'mean:{mean_obj_occlusion * 100:.2f} [%], max:{max_obj_occlusion * 100:.2f} [%]\t\t'
                  f'{aff_name}'
                  f'mean:{mean_obj_part_occlusion * 100:.2f} [%], max:{max_obj_part_occlusion * 100:.2f} [%]',
                  )

if __name__ == '__main__':
    # run all test.
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(TestARLAffPoseDataloader("load_gt_pose"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
