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

from tools.utils import helper_utils

from tools.YCB import cfg as config
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

        rgb_addr = config.DATASET_ROOT_PATH + image_addr + config.RGB_EXT
        depth_addr = config.DATASET_ROOT_PATH + image_addr + config.DEPTH_EXT
        label_addr = config.DATASET_ROOT_PATH + image_addr + config.LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        label = np.array(Image.open(label_addr))

        # gt pose
        meta_addr = config.DATASET_ROOT_PATH + image_addr + config.META_EXT
        meta = scio.loadmat(meta_addr)

        cv2_bbox_img = rgb.copy()
        cv2_obj_img = rgb.copy()

        #######################################
        # project to screen
        #######################################

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        for idx in range(len(obj)):
            obj_idx = obj[idx]

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj_idx))
            mask = mask_label * mask_depth

            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            cv2_bbox_img = np.transpose(np.array(cv2_bbox_img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

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

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > config.NUM_PT:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:config.NUM_PT] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, config.NUM_PT - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = config.XMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = config.YMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = meta['factor_depth'][0][0]
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            dellist = [j for j in range(0, len(cld[obj_idx]))]
            dellist = random.sample(dellist, len(cld[obj_idx]) - config.NUM_PT_MESH_SMALL)
            model_points = np.delete(cld[obj_idx], dellist, axis=0)

            #######################################
            #######################################

            target_r = meta['poses'][:, :, idx][:, 0:3]
            target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])

            #######################################
            # PROJECT TO SCREEN
            #######################################
            obj_color = ycb_dataset_utils.obj_color_map(obj_idx)

            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            ### raw point cloud
            imgpts, jac = cv2.projectPoints(cld[obj_idx]*1e3, target_r, target_t* 1e3, cam_mat, cam_dist)
            cv2_obj_img = cv2.polylines(np.array(cv2_obj_img), np.int32([np.squeeze(imgpts)]), True, obj_color)

            ### cloud
            # imgpts, jac = cv2.projectPoints(cloud * 1e3, np.eye(3), np.zeros(shape=target_t.shape), cam_mat, cam_dist)
            # cv2_obj_img = cv2.polylines(np.array(cv2_obj_img), np.int32([np.squeeze(imgpts)]), True, obj_color)

            ### model_points
            # imgpts, jac = cv2.projectPoints(model_points * 1e3, target_r, target_t * 1e3, cam_mat, cam_dist)
            # cv2_obj_img = cv2.polylines(np.array(cv2_obj_img), np.int32([np.squeeze(imgpts)]), True, obj_color)

            # draw model
            cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, obj_color)
            # draw bbox
            cv2_obj_img = cv2.rectangle(cv2_obj_img, (cmin, rmin), (cmax, rmax), obj_color, 2)
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
        label = cv2.resize(label, config.RESIZE)
        cv2_obj_img = cv2.resize(cv2_obj_img, config.RESIZE)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('label', ycb_dataset_utils.colorize_obj_mask(label))
        cv2.imshow('cv2_obj_img', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()