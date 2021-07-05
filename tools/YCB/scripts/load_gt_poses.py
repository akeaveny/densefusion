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

from tools.YCB import cfg as config

from tools.YCB.utils import helper_utils
from tools.YCB.utils.dataset import ycb_dataset_utils

from tools.YCB.utils.pose.load_obj_ply_files import load_obj_ply_files

from tools.YCB.utils.bbox.extract_bboxs_from_label import get_bbox, get_obj_bbox

#######################################
#######################################

def main():

    ###################################
    # Load Ply files
    ###################################

    cld, obj_classes, obj_class_ids = load_obj_ply_files()

    ##################################
    ##################################

    image_files = open('{}'.format(config.TRAIN_FILE), "r")
    # image_files = open('{}'.format(config.TEST_FILE), "r")
    image_files = image_files.readlines()
    print("Loaded Files: {}".format(len(image_files)))

    # select random test images
    np.random.seed(0)
    num_files = 250
    # random_idx = np.random.choice(np.arange(0, int(16187), 1), size=int(num_files), replace=False) # ONLY REAL
    # random_idx = np.random.choice(np.arange(int(16187+1), int(len(image_files)), 1), size=int(num_files), replace=False) # ONLY SYN
    random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    image_files = np.array(image_files)[random_idx]
    print("Chosen Files: {}".format(len(image_files)))

    # image_files = np.array(image_files)[:16188]  # REAL
    # image_files = np.array(image_files)[16188+1:] # SYN
    # print("Chosen Files: {}".format(len(image_files)))

    for image_idx, image_addr in enumerate(image_files):

        image_addr = image_addr.rstrip()
        print(f'\nimage:{image_idx}/{len(image_files)}, image_addr:{image_addr} ..')

        rgb_addr = config.DATASET_ROOT_PATH + image_addr + config.RGB_EXT
        depth_addr = config.DATASET_ROOT_PATH + image_addr + config.DEPTH_EXT
        label_addr = config.DATASET_ROOT_PATH + image_addr + config.LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        label = np.array(Image.open(label_addr))

        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]

        #######################################
        #######################################

        color_label = ycb_dataset_utils.colorize_obj_mask(label)
        color_label = cv2.addWeighted(rgb, 0.35, color_label, 0.65, 0)

        cv2_bbox_img = color_label.copy()
        cv2_obj_img = color_label.copy()

        #######################################
        #######################################

        # gt pose
        meta_addr = config.DATASET_ROOT_PATH + image_addr + config.META_EXT
        meta = scio.loadmat(meta_addr)

        #######################################
        # project to screen
        #######################################

        obj_cls_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        label_class_indxs = np.unique(label)[1:]
        print(f'label_class_indxs:{label_class_indxs}')

        for idx in range(len(obj_cls_idxs)):
            obj_id = obj_cls_idxs[idx]
            if obj_id in label_class_indxs:
                print(f"\tObject:", obj_classes[int(obj_id) - 1])
                obj_color = ycb_dataset_utils.obj_color_map(obj_id)

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

                cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

                #######################################
                # 6-DoF Pose
                #######################################

                target_r = np.array(meta['poses'][:, :, idx][:, 0:3], dtype=np.float64)
                target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()], dtype=np.float64)

                print(f'target_r: {target_r.dtype}')
                print(f'target_t: {target_t.dtype}')

                #######################################
                # MASK
                #######################################

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, obj_id))
                mask = mask_label * mask_depth

                #######################################
                # BBOX
                #######################################

                rmin, rmax, cmin, cmax = get_bbox(mask_label)
                # cv2_bbox_img = np.transpose(np.array(cv2_bbox_img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

                # draw bbox
                cv2_obj_img = cv2.rectangle(cv2_obj_img, (cmin, rmin), (cmax, rmax), obj_color, 2)

                # draw object name
                cv2_obj_img = cv2.putText(cv2_obj_img,
                                          ycb_dataset_utils.map_obj_id_to_name(obj_id),
                                          (cmin, rmin - 5),
                                          cv2.FONT_ITALIC,
                                          0.4,
                                          (255, 255, 255))

                #######################################
                # OBJ: 6-DoF Pose
                #######################################
                obj_color = ycb_dataset_utils.obj_color_map(obj_id)

                # project points
                imgpts, jac = cv2.projectPoints(cld[obj_id] * 1e3, target_r, target_t * 1e3, cam_mat, cam_dist)
                # cv2_obj_img = cv2.polylines(np.array(cv2_obj_img), np.int32([np.squeeze(imgpts)]), True, obj_color)

                # draw pose
                rotV, _ = cv2.Rodrigues(target_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, target_t * 1e3, cam_mat, cam_dist)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

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
        color_label = cv2.resize(color_label, config.RESIZE)
        cv2_obj_img = cv2.resize(cv2_obj_img, config.RESIZE)

        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        # cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        # cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('cv2_obj_img', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(1)

if __name__ == '__main__':
    main()