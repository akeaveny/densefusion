import glob
import copy
import random

import numpy as np
import numpy.ma as ma

import cv2
from PIL import Image

import scipy.io as scio

import argparse

#######################################
#######################################

from tools.Elevator.utils import helper_utils

from tools.Elevator import cfg as config
from tools.Elevator.utils.dataset import elavator_dataset_utils

from tools.Elevator.utils.pose.load_obj_ply_files import load_obj_ply_files
from tools.Elevator.utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

def main():

    ###################################
    # Load Ply files
    ###################################

    cld, obj_classes, obj_ids = load_obj_ply_files()

    ##################################
    ##################################

    # image_files = open('{}'.format(config.TRAIN_FILE), "r")
    # image_files = open('{}'.format(config.VAL_FILE), "r")
    image_files = open('{}'.format(config.TEST_FILE), "r")
    image_files = image_files.readlines()
    print("Loaded Files: {}".format(len(image_files)))

    # select random test images
    # np.random.seed(0)
    # num_files = 25
    # random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    # image_files = np.array(image_files)[random_idx]
    print("Chosen Files: {}".format(len(image_files)))
    good_choose_files = []

    for image_idx, image_addr in enumerate(image_files):

        ##################################
        # init
        ##################################

        image_addr = image_addr.rstrip()
        dataset_dir = image_addr.split('rgb/')[0]
        image_num = image_addr.split('rgb/')[-1]

        rgb_addr   = dataset_dir + 'rgb/'   + image_num + config.RGB_EXT
        depth_addr = dataset_dir + 'depth/' + image_num + config.DEPTH_EXT
        label_addr = dataset_dir + 'masks/' + image_num + config.LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        label = np.array(Image.open(label_addr))

        ##################################
        ### RESIZE & CROP
        ##################################

        rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        rgb = helper_utils.crop(pil_img=rgb, crop_size=config.INPUT_SIZE, is_img=True)
        label = helper_utils.crop(pil_img=label, crop_size=config.INPUT_SIZE)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.INPUT_SIZE)

        ##################################
        # META
        ##################################

        # gt pose
        meta_addr = dataset_dir + 'meta/' + image_num + config.META_EXT
        meta = scio.loadmat(meta_addr)

        cv2_bbox_img = rgb.copy()
        cv2_obj_img = rgb.copy()

        #####################
        #####################

        obj_ids = np.array(meta['object_class_ids']).flatten()
        label_obj_ids = np.unique(label)[1:]

        #####################
        #####################

        for idx, obj_id in enumerate(obj_ids):
            if obj_id in label_obj_ids:
                print("Object:", obj_classes[int(obj_id) - 1])
                obj_color = elavator_dataset_utils.obj_color_map(obj_id)

                ####################
                # GT POSE
                ####################

                obj_meta_idx = str(1000 + obj_id)[1:]
                target_r = meta['obj_rotation_' + np.str(obj_meta_idx)]
                target_t = meta['obj_translation_' + np.str(obj_meta_idx)]

                # projecting 3D model to 2D image
                imgpts, jac = cv2.projectPoints(cld[obj_id] * 1e3, target_r, target_t * 1e3, config.CAM_MAT,
                                                config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                # draw pose
                rotV, _ = cv2.Rodrigues(target_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                ##################################
                # BBOX
                ##################################

                mask_label = ma.getmaskarray(ma.masked_equal(label, obj_id))
                mask_rgb = np.repeat(mask_label, 3).reshape(label.shape[0], label.shape[1], -1) * rgb
                # mask_depth = mask_label * ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_depth = mask_label * depth

                # obj_bbox = np.array(meta['obj_bbox_' + np.str(obj_meta_idx)]).flatten()
                # x1, y1, x2, y2 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
                x1, y1, x2, y2 = get_obj_bbox(mask_label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)

                # drawing bbox = (x1, y1), (x2, y2)
                cv2_obj_img = cv2.rectangle(cv2_obj_img, (x1, y1), (x2, y2), obj_color, 2)

                cv2_obj_img = cv2.putText(cv2_obj_img,
                                          elavator_dataset_utils.map_obj_id_to_name(obj_id),
                                          (x1, y1 - 5),
                                          cv2.FONT_ITALIC,
                                          0.4,
                                          obj_color)

                ##################################
                # CHOOSE
                ##################################

                choose = mask_depth[y1:y2, x1:x2].flatten().nonzero()[0]
                print("\tchoose: ", len(choose))
                if len(choose) != 0:
                    good_choose_files.append(image_addr)

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

        # color_label = elavator_dataset_utils.colorize_obj_mask(label)
        #
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # cv2.imshow('depth', depth)
        # cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        # cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        # cv2.imshow('gt_pose', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))
        #
        # cv2.waitKey(0)

    #####################
    #####################

    print("Good Files: {}".format(len(good_choose_files)))
    # f_train = open(config.TRAIN_FILE, 'w')
    # f_train = open(config.VAL_FILE, 'w')
    f_train = open(config.TEST_FILE, 'w')
    # ===================== train ====================
    for i, file in enumerate(good_choose_files):
        # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TRAIN + 'rgb/')[1]
        str_num = file.split(config.RGB_EXT)[0]
        f_train.write(str_num)
        f_train.write('\n')
    print('wrote {} files'.format(i + 1))

if __name__ == '__main__':
    main()