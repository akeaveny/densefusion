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

import sys
sys.path.append('../../..')

#######################################
#######################################

from tools.ARLAffPose.utils import helper_utils

from tools.ARLAffPose import cfg as config
from tools.ARLAffPose.utils.dataset import affpose_dataset_utils

from tools.ARLAffPose.utils.pose.load_obj_ply_files import load_obj_ply_files
from tools.ARLAffPose.utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

def main():

    ###################################
    # Load Ply files
    ###################################

    cld, cld_obj_centered, cld_obj_part_centered, obj_classes, obj_part_classes, obj_ids, obj_part_ids = load_obj_ply_files()

    ##################################
    ##################################

    # image_files = open('{}'.format(config.TRAIN_FILE), "r")
    image_files = open('{}'.format(config.VAL_FILE), "r")
    # image_files = open('{}'.format(config.TEST_FILE), "r")
    image_files = image_files.readlines()
    print("Loaded Files: {}".format(len(image_files)))

    # select random test images
    np.random.seed(0)
    num_files = 25
    random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    image_files = np.array(image_files)[random_idx]
    print("Chosen Files: {}".format(len(image_files)))
    good_choose_files = []

    for image_idx, image_addr in enumerate(image_files):

        ##################################
        # init
        ##################################

        image_addr = image_addr.rstrip()
        dataset_dir = image_addr.split('rgb/')[0]
        image_num = image_addr.split('rgb/')[-1]

        print('\nimage:{}/{}, file:{}'.format(image_idx + 1, len(image_files), image_addr))

        rgb_addr   = dataset_dir + 'rgb/'   + image_num + config.RGB_EXT
        depth_addr = dataset_dir + 'depth/' + image_num + config.DEPTH_EXT
        obj_label_addr = dataset_dir + 'masks_obj/' + image_num + config.OBJ_LABEL_EXT
        obj_part_label_addr = dataset_dir + 'masks_obj_part/' + image_num + config.OBJ_PART_LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))[..., :3]
        depth = np.array(Image.open(depth_addr))
        obj_label = np.array(Image.open(obj_label_addr))
        obj_part_label = np.array(Image.open(obj_part_label_addr))

        ##################################
        ### RESIZE & CROP
        ##################################

        rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_label = cv2.resize(obj_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_part_label = cv2.resize(obj_part_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        rgb = helper_utils.crop(pil_img=rgb, crop_size=config.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        obj_label = helper_utils.crop(pil_img=obj_label, crop_size=config.CROP_SIZE)
        obj_part_label = helper_utils.crop(pil_img=obj_part_label, crop_size=config.CROP_SIZE)

        #####################
        #####################

        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # cv2.imshow('obj_label', cv2.cvtColor(affpose_dataset_utils.colorize_obj_mask(obj_label), cv2.COLOR_BGR2RGB))
        # cv2.imshow('mask',   ma.getmaskarray(ma.masked_equal(obj_label, 7)).astype(np.uint8)*50)
        #
        # cv2.waitKey(0)

        #####################
        #####################

        cv2_obj_img = rgb.copy()
        cv2_obj_parts_img = rgb.copy()

        ##################################
        # META
        ##################################

        # gt pose
        meta_addr = dataset_dir + 'meta/' + image_num + config.META_EXT
        meta = scio.loadmat(meta_addr)

        #####################
        #####################

        obj_ids = np.array(meta['object_class_ids']).flatten()
        label_obj_ids = np.unique(obj_label)[1:]
        print("label_obj_ids: ", label_obj_ids)

        #######################################
        #######################################

        for idx, obj_id in enumerate(obj_ids):
            print("Object:", obj_classes[int(obj_id) - 1])

            #######################################
            # OBJECT
            #######################################
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)

            obj_meta_idx = str(1000 + obj_id)[1:]
            obj_r = meta['obj_rotation_' + np.str(obj_meta_idx)]
            obj_t = meta['obj_translation_' + np.str(obj_meta_idx)]

            #######################################
            # BBOX
            #######################################

            x1, y1, x2, y2 = get_obj_bbox(obj_label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)

            # drawing bbox = (x1, y1), (x2, y2)
            cv2_obj_img = cv2.rectangle(cv2_obj_img, (x1, y1), (x2, y2), obj_color, 2)

            cv2_obj_img = cv2.putText(cv2_obj_img,
                                      affpose_dataset_utils.map_obj_id_to_name(obj_id),
                                      (x1, y1 - 5),
                                      cv2.FONT_ITALIC,
                                      0.4,
                                      obj_color)

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            print(f'obj_part_ids:{obj_part_ids}')
            for obj_part_id in obj_part_ids:
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                print(f"\tAff: {aff_id}, {obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # 6-DOF POSE
                #######################################
                obj_centered = cld_obj_centered[obj_part_id]

                # projecting 3D model to 2D image
                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                # modify YCB objects rotation matrix
                _obj_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_r.copy())

                # draw pose
                rotV, _ = cv2.Rodrigues(_obj_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                #######################################
                # OBJECT PART AFF CENTERED
                #######################################
                aff_color = affpose_dataset_utils.aff_color_map(aff_id)

                obj_part_centered = cld_obj_part_centered[obj_part_id]
                obj_part_id_idx = str(1000 + obj_part_id)[1:]
                obj_part_r = meta['obj_part_rotation_' + np.str(obj_part_id_idx)]
                obj_part_t = meta['obj_part_translation_' + np.str(obj_part_id_idx)]

                #######################################
                # BBOX
                #######################################

                obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_bbox(obj_part_label.copy(), obj_part_id,
                                                                                  config.HEIGHT, config.WIDTH,
                                                                                  config.BORDER_LIST)

                # drawing bbox = (x1, y1), (x2, y2)
                # cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # white
                # cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), aff_color, 2)
                #
                # cv2_obj_parts_img = cv2.putText(cv2_obj_parts_img,
                #                                 affpose_dataset_utils.map_obj_id_to_name(obj_id),
                #                                 (x1, y1 - 5),
                #                                 cv2.FONT_ITALIC,
                #                                 0.4,
                #                                 (255, 255, 255))  # red

                #######################################
                # 6-DOF POSE
                #######################################

                # draw model
                obj_parts_imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_parts_img = cv2.polylines(cv2_obj_parts_img, np.int32([np.squeeze(obj_parts_imgpts)]), False, aff_color)

                if obj_part_id in affpose_dataset_utils.DRAW_OBJ_PART_POSE:
                    # modify YCB objects rotation matrix
                    _obj_part_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_part_r.copy())
                    # draw pose
                    rotV, _ = cv2.Rodrigues(_obj_part_r)
                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                    axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                ##################################
                # CHOOSE
                ##################################

                mask_label = ma.getmaskarray(ma.masked_equal(obj_part_label.copy(), obj_part_id))
                mask_depth = mask_label * depth

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

        helper_utils.print_class_labels(obj_label)

        #####################
        # PLOTTING
        #####################

        color_obj_label = affpose_dataset_utils.colorize_obj_mask(obj_label)
        color_obj_part_label = affpose_dataset_utils.colorize_obj_mask(obj_part_label)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('obj_label', cv2.cvtColor(color_obj_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('obj_part_label', cv2.cvtColor(color_obj_part_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('gt_obj_pose', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))
        cv2.imshow('gt_obj_part_pose', cv2.cvtColor(cv2_obj_parts_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

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