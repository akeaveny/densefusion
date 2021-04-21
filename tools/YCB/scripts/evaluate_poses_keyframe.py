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
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable

#######################################
#######################################

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

#######################################
#######################################

from tools.utils import helper_utils

from tools.YCB import cfg as config
from tools.YCB.utils.dataset import ycb_dataset_utils

from tools.YCB.utils.pose.load_obj_ply_files import load_obj_ply_files

from tools.YCB.utils.bbox.extract_bboxs_from_label import get_bbox, get_obj_bbox, get_posecnn_bbox

#######################################
#######################################

def main():

    ###################################
    # Load Ply files
    ###################################

    cld, obj_classes, obj_class_ids = load_obj_ply_files()

    ##################################
    ## DENSEFUSION
    ##################################

    estimator = PoseNet(num_points=config.NUM_PT, num_obj=config.NUM_OBJECTS)
    estimator.cuda()
    estimator.load_state_dict(torch.load(config.PRE_TRAINED_MODEL))
    estimator.eval()

    refiner = PoseRefineNet(num_points=config.NUM_PT, num_obj=config.NUM_OBJECTS)
    refiner.cuda()
    refiner.load_state_dict(torch.load(config.PRE_TRAINED_REFINE_MODEL))
    refiner.eval()

    ##################################
    ##################################

    # image_files = open('{}'.format(config.TRAIN_FILE), "r")
    image_files = open('{}'.format(config.TEST_FILE), "r")
    image_files = image_files.readlines()
    print("Loaded Files: {}".format(len(image_files)))

    ### select subset of images
    # num_files = 25
    # idx = np.arange(0, int(num_files))
    # image_files = np.array(image_files)[idx]
    # print("Chosen Files: {}".format(len(image_files)))

    ##################################
    ##################################

    for image_idx, image_addr in enumerate(image_files):

        image_addr = 'data/' + image_addr.rstrip()

        rgb_addr = config.DATASET_ROOT_PATH + image_addr + config.RGB_EXT
        depth_addr = config.DATASET_ROOT_PATH + image_addr + config.DEPTH_EXT
        label_addr = config.DATASET_ROOT_PATH + image_addr + config.LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        label = np.array(Image.open(label_addr))

        # gt pose
        meta_addr = config.DATASET_ROOT_PATH + image_addr + config.META_EXT
        meta = scio.loadmat(meta_addr)

        #######################################
        #######################################

        # posecnn
        posecnn_meta_idx = str(1000000 + image_idx)[1:] # gt results and posecnn are offset by 1
        posecnn_meta_addr = config.YCB_TOOLBOX_CONFIG + posecnn_meta_idx + config.POSECNN_EXT
        posecnn_meta = scio.loadmat(posecnn_meta_addr)

        posecnn_label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])
        poses_icp = np.array(posecnn_meta['poses_icp'])

        pred_cls_idxs = np.array(posecnn_rois[:, 1], dtype=int)

        gt_cls_idxs = np.array(meta['cls_indexes'].flatten(), dtype=int)
        gt_poses = np.array(meta['poses']).flatten().reshape(3, 4, -1)

        pred_to_gt_idx = []
        for value in pred_cls_idxs:
            if value in gt_cls_idxs.tolist():
                pred_to_gt_idx.append(gt_cls_idxs.tolist().index(value))

        print("\npred_cls_idxs: {}".format(pred_cls_idxs))
        print("gt_cls_idxs: {}".format(gt_cls_idxs))
        print("posecnn_roi_idxs: {}".format(pred_to_gt_idx))

        #######################################
        #######################################

        cv2_bbox_img = rgb.copy()
        cv2_gt_pose = rgb.copy()
        cv2_pose_cnn = rgb.copy()
        cv2_densefusion = rgb.copy()

        #######################################
        #######################################

        class_ids_list = []
        pose_est_gt = []
        pose_est_posecnn = []
        pose_est_df_wo_refine = []
        pose_est_df_iterative = []

        for class_idx in range(len(pred_cls_idxs)):

            pred_cls_idx = pred_cls_idxs[class_idx]
            class_ids_list.append(pred_cls_idx)
            print("\n*** {} ***".format(obj_classes[int(pred_cls_idx) - 1]))

            # gt_idx = pred_to_gt_idx[class_idx]
            # print("PoseCNN RoI Idx:{}, GT Idx:{}".format(gt_idx, class_idx))
            # print("PoseCNN:{}, GT:{}".format(gt_cls_idxs[gt_idx], pred_cls_idx))

            ############################
            # pose_cnn
            ############################

            # posecnn
            pose_cnn_pose = poses_icp[class_idx, :]
            pose_est_posecnn.append(np.array(pose_cnn_pose).tolist())

            try:

                #######################################
                # bbox
                #######################################

                rmin, rmax, cmin, cmax = get_posecnn_bbox(posecnn_rois, class_idx)

                #######################################
                # real cam for test frames
                #######################################
                cam_cx = config.CAM_CX_1
                cam_cy = config.CAM_CY_1
                cam_fx = config.CAM_FX_1
                cam_fy = config.CAM_FY_1

                #######################################
                #######################################

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, pred_cls_idx))
                mask = mask_label * mask_depth

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

                pt2 = depth_masked / config.CAM_SCALE
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(rgb)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = config.NORM(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([pred_cls_idx - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, config.NUM_PT, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                #######################################
                #######################################

                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, config.NUM_PT, 1)

                pred_c = pred_c.view(config.BATCH_SIZE, config.NUM_PT)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(config.BATCH_SIZE * config.NUM_PT, 1, 3)
                points = cloud.view(config.BATCH_SIZE * config.NUM_PT, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                pose_est_df_wo_refine.append(my_pred.tolist())

                for ite in range(0, config.REFINE_ITERATIONS):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(config.NUM_PT, 1).contiguous().view(1, config.NUM_PT, 3)
                    my_mat = quaternion_matrix(my_r)
                    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                    my_mat[0:3, 3] = my_t

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = refiner(new_cloud, emb, index)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0:3, 3] = my_t_2

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final

                pose_est_df_iterative.append(my_pred.tolist())

                # #######################################
                # # PROJECT TO SCREEN
                # #######################################
                #
                # obj_color = ycb_dataset_utils.obj_color_map(pred_cls_idx)
                #
                # cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                # cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                #
                # #######################################
                # # gt
                # #######################################
                #
                # gt_pose = gt_poses[:, :, gt_idx]
                # gt_rot = gt_pose[0:3, 0:3]
                # gt_trans = gt_pose[0:3, -1]
                #
                # gt_quart = quaternion_from_matrix(gt_rot)
                # my_pred = np.append(np.array(gt_quart), np.array(gt_trans))
                # pose_est_gt.append(my_pred.tolist())
                #
                # ### raw point cloud
                # imgpts, jac = cv2.projectPoints(cld[pred_cls_idx], gt_rot, gt_trans, cam_mat, cam_dist)
                # cv2_gt_pose = cv2.polylines(cv2_gt_pose, np.int32([np.squeeze(imgpts)]), True, obj_color)
                #
                # # draw bbox
                # cv2_gt_pose = cv2.rectangle(cv2_gt_pose, (cmin, rmin), (cmax, rmax), obj_color, 2)
                #
                # # posecnn
                # imgpts, jac = cv2.projectPoints(cld[pred_cls_idx], gt_rot, gt_trans, cam_mat, cam_dist)
                # cv2_pose_cnn = cv2.polylines(cv2_pose_cnn, np.int32([np.squeeze(imgpts)]), True, obj_color)
                # cv2_pose_cnn = cv2.rectangle(cv2_pose_cnn, (cmin, rmin), (cmax, rmax), obj_color, 2)
                #
                # # densefusion
                # imgpts, jac = cv2.projectPoints(cld[pred_cls_idx], gt_rot, gt_trans, cam_mat, cam_dist)
                # cv2_densefusion = cv2.polylines(cv2_densefusion, np.int32([np.squeeze(imgpts)]), True, obj_color)
                # cv2_densefusion = cv2.rectangle(cv2_densefusion, (cmin, rmin), (cmax, rmax), obj_color, 2)
                #
                # ############################
                # # pose_cnn
                # ############################
                #
                # pose_cnn_rot = quaternion_matrix(pose_cnn_pose[0:4])[0:3, 0:3]
                # pose_cnn_trans = pose_cnn_pose[4:7]
                #
                # # selecting subset of cloud for cleaner projections
                # cld_idx = np.random.choice(np.arange(0, int(len(cld[pred_cls_idx])), 1), size=config.NUM_PT,
                #                            replace=False)
                # cloud = cld[pred_cls_idx][cld_idx]
                #
                # imgpts_gt, jac = cv2.projectPoints(cloud, pose_cnn_rot, pose_cnn_trans, cam_mat, cam_dist)
                # cv2_pose_cnn = cv2.polylines(cv2_pose_cnn, np.int32([np.squeeze(imgpts_gt)]), True,
                #                              ycb_dataset_utils.pose_cnn_pred_color())
                #
                # ############################
                # # pred
                # ############################
                #
                # mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                # my_t = my_t
                #
                # # selecting subset of cloud for cleaner projections
                # cld_idx = np.random.choice(np.arange(0, int(len(cld[pred_cls_idx])), 1), size=config.NUM_PT, replace=False)
                # cloud = cld[pred_cls_idx][cld_idx]
                #
                # # pred
                # imgpts, jac = cv2.projectPoints(cloud, mat_r, my_t, cam_mat, cam_dist)
                # cv2_densefusion = cv2.polylines(cv2_densefusion, np.int32([np.squeeze(imgpts)]), True,
                #                                 ycb_dataset_utils.densefusion_pred_color())

            except ZeroDivisionError:
                print("DenseFusion Detector Lost {0} at No.{1} keyframe".format(class_idx, class_idx))
                pose_est_df_wo_refine.append([0.0 for i in range(7)])
                pose_est_df_iterative.append([0.0 for i in range(7)])

        ############################
        ### PLOTTING
        ############################

        # cv2.imshow('cv2_gt_pose', cv2.cvtColor(cv2_gt_pose, cv2.COLOR_BGR2RGB))
        # cv2.imshow('cv2_pose_cnn', cv2.cvtColor(cv2_pose_cnn, cv2.COLOR_BGR2RGB))
        # cv2.imshow('cv2_densefusion', cv2.cvtColor(cv2_densefusion, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)

        ############################
        ############################

        scio.savemat('{0}/{1}.mat'.format(config.EVAL_FOLDER_GT, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_gt})
        scio.savemat('{0}/{1}.mat'.format(config.EVAL_FOLDER_POSECNN, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_posecnn})
        scio.savemat('{0}/{1}.mat'.format(config.EVAL_FOLDER_DF_WO_REFINE, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_df_wo_refine})
        scio.savemat('{0}/{1}.mat'.format(config.EVAL_FOLDER_DF_ITERATIVE, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_df_iterative})

        print("******************* Finish {0}/{1} keyframes *******************\n".format(image_idx+1, len(image_files)))


if __name__ == '__main__':
    main()