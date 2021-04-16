import os
import glob
import copy
import random

import numpy as np
import numpy.ma as ma

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as scio

import argparse

#######################################
#######################################

import tools.YCB_Aff.cfg as config

from tools.YCB_Aff.utils import helper_utils
from tools.YCB_Aff.utils.dataset import ycb_aff_dataset_utils

from tools.YCB_Aff.utils.pose.load_obj_part_ply_files import load_obj_part_ply_files

from tools.YCB_Aff.utils.bbox.extract_bboxs_from_label import get_bbox, get_obj_part_bbox

#######################################
#######################################

def main():

    ###################################
    # Load Ply files
    ###################################

    cld, cld_obj_centered, cld_obj_part_centered, obj_classes, obj_part_classes = load_obj_part_ply_files()

    ##################################
    ##################################

    image_files = open('{}'.format(config.TRAIN_FILE), "r")
    # image_files = open('{}'.format(config.TEST_FILE), "r")
    image_files = image_files.readlines()
    print("Loaded Files: {}".format(len(image_files)))

    # select random test images
    np.random.seed(1)
    num_files = 25
    random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    image_files = np.array(image_files)[random_idx]
    print("Chosen Files: {}".format(len(image_files)))

    for image_idx, image_addr in enumerate(image_files):

        image_addr = image_addr.rstrip()
        print('\n{}/{}, image_addr:{}'.format(image_idx+1, len(image_files), image_addr))

        rgb_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.RGB_EXT
        depth_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.DEPTH_EXT
        label_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.AFF_LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        aff_label = np.array(Image.open(label_addr))

        # gt pose
        meta_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.META_EXT
        meta = scio.loadmat(meta_addr)

        cv2_obj_parts_img = rgb.copy()

        #######################################
        #######################################

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        for idx in range(len(obj)):
            obj_id = obj[idx]
            print("Object:", obj_classes[int(obj_id) - 1])

            #######################################
            #######################################

            if image_addr.split('/')[0] != 'data_syn' and int(image_addr.split('/')[1]) >= 60:
                cam_cx = config.CAM_CX_2
                cam_cy = config.CAM_CY_2
                cam_fx = config.CAM_FX_2
                cam_fy = config.CAM_FY_2
            else:
                cam_cx = config.CAM_CX_1
                cam_cy = config.CAM_CY_1
                cam_fx = config.CAM_FX_1
                cam_fy = config.CAM_FY_1

            #######################################
            #######################################

            obj_r = meta['poses'][:, :, idx][:, 0:3]
            obj_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])

            obj_meta_idx = str(1000 + obj_id)[1:]
            #  cmin, rmin, cmax, rmax
            obj_bbox = meta['obj_bbox_' + np.str(obj_meta_idx)].flatten()
            cmin, rmin, cmax, rmax = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]

            #######################################
            # PROJECT TO SCREEN
            #######################################

            obj_color = ycb_aff_dataset_utils.obj_color_map(obj_id)

            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = ycb_aff_dataset_utils.map_obj_ids_to_obj_part_ids(obj_id)
            print(f'obj_part_ids:{obj_part_ids}')
            for obj_part_id in obj_part_ids:
                aff_id = ycb_aff_dataset_utils.map_obj_part_ids_to_aff_ids(obj_part_id)
                aff_color = ycb_aff_dataset_utils.aff_color_map(aff_id)
                print(f"\tAff: {aff_id}, {obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # meta
                #######################################
                obj_part_id_idx = str(1000 + obj_part_id)[1:]

                obj_part_bbox = meta['obj_part_bbox_' + np.str(obj_part_id_idx)].flatten()
                x1, y1, x2, y2 = obj_part_bbox[0], obj_part_bbox[1], obj_part_bbox[2], obj_part_bbox[3]

                obj_part_r = meta['obj_part_rot_' + np.str(obj_part_id_idx)]
                obj_part_t = meta['obj_part_trans__' + np.str(obj_part_id_idx)]

                #######################################
                #######################################
                # if aff_id == 2:

                # draw model
                aff_imgpts, jac = cv2.projectPoints(cld_obj_part_centered[obj_part_id] * 1e3, obj_part_r, obj_part_t * 1e3, cam_mat, cam_dist)
                cv2_obj_parts_img = cv2.polylines(cv2_obj_parts_img, np.int32([np.squeeze(aff_imgpts)]), False, aff_color)

                # # drawing bbox = (x1, y1), (x2, y2) = (cmin, rmin), (cmax, rmax)
                cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (x1, y1), (x2, y2), aff_color, 2)
                cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

                cv2_obj_parts_img = cv2.putText(cv2_obj_parts_img,
                                                ycb_aff_dataset_utils.map_obj_id_to_name(obj_id),
                                                (cmin, rmin - 5),
                                                cv2.FONT_ITALIC,
                                                0.4,
                                                (255, 0, 0))

                # draw pose
                # rotV, _ = cv2.Rodrigues(obj_part_r)
                # points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                # axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, cam_mat, cam_dist)
                # cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0, 0, 255), 3)
                # cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                # cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 0), 3)

        #####################
        # DEPTH INFO
        #####################

        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        #####################
        # LABEL INFO
        #####################

        helper_utils.print_class_labels(aff_label)

        #####################
        # PLOTTING
        #####################

        rgb = cv2.resize(rgb, config.RESIZE)
        depth = cv2.resize(depth, config.RESIZE)
        aff_label = cv2.resize(aff_label, config.RESIZE)
        color_aff_label = ycb_aff_dataset_utils.colorize_aff_mask(aff_label)
        cv2_obj_parts_img = cv2.resize(cv2_obj_parts_img, config.RESIZE)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('cv2_obj_parts_img', cv2.cvtColor(cv2_obj_parts_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()