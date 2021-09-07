import os
import glob

import copy
import random
import time

import numpy as np
import numpy.ma as ma

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as scio
from scipy.spatial.transform import Rotation as R

from sklearn.neighbors import KDTree

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

import sys
sys.path.append('../../../')

#######################################
#######################################

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

#######################################
#######################################

from affpose.YCB_Aff import cfg as config
from affpose.YCB_Aff.dataset import ycb_aff_dataset_utils
from affpose.YCB_Aff.dataset import dataloader as ycb_aff_dataloader
from affpose.YCB_Aff.utils.bbox.extract_bboxs_from_label import get_obj_part_bbox, get_posecnn_bbox
from affpose.YCB_Aff.eval import eval_utils

#######################################
#######################################

DELETE_OLD_RESULTS = True

SPLIT = 'test'
SELECT_RANDOM_IMAGES = False
NUM_IMAGES = 100

USE_PRED_MASKS = False

VISUALIZE_AND_GET_ERROR_METRICS = False
PROJECT_MESH_ON_IMAGE = False


def main():

    ##################################
    # Remove old results.
    ##################################

    if DELETE_OLD_RESULTS:
        files = glob.glob(config.OBJ_EVAL_FOLDER_GT + '/*')
        for file in files:
            os.remove(file)

        files = glob.glob(config.OBJ_EVAL_FOLDER_DF_WO_REFINE + '/*')
        for file in files:
            os.remove(file)

        files = glob.glob(config.OBJ_EVAL_FOLDER_DF_ITERATIVE + '/*')
        for file in files:
            os.remove(file)

    ##################################
    # DENSEFUSION
    ##################################

    estimator = PoseNet(num_points=config.NUM_PT, num_obj=config.NUM_OBJECTS)
    estimator.cuda()
    estimator.load_state_dict(torch.load(config.PRE_TRAINED_MODEL))
    estimator.eval()

    refiner = PoseRefineNet(num_points=config.NUM_PT, num_obj=config.NUM_OBJECTS)
    refiner.cuda()
    refiner.load_state_dict(torch.load(config.PRE_TRAINED_REFINE_MODEL))
    refiner.eval()

    img_norm = transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)

    ###################################
    # Load ARL AFFPose
    ###################################

    # load real images.
    dataloader = ycb_aff_dataloader.YCBAff(split=SPLIT,
                                           select_random_images=SELECT_RANDOM_IMAGES,
                                           num_images=NUM_IMAGES)

    ###################################
    # Stats
    ###################################

    stats_pred_class_ids = np.zeros(shape=(len(dataloader.img_files), 10))
    stats_pred_choose = np.zeros(shape=(len(dataloader.img_files), 10))
    stats_pred_c = np.zeros(shape=(len(dataloader.img_files), 10))

    for image_idx, image_addr in enumerate(dataloader.img_files):
        t0 = time.time()

        #####################
        # Load GT images.
        #####################

        data = dataloader.draw_gt_obj_pose(image_idx, project_mesh_on_image=PROJECT_MESH_ON_IMAGE)

        rgb = data["rgb"]
        depth_16bit = data["depth_16bit"]
        depth_8bit = data["depth_8bit"]
        obj_label = data["obj_label"]
        obj_part_label = data["obj_part_label"]
        aff_label = data["aff_label"]
        cv2_obj_pose_img = data["cv2_obj_pose_img"]
        cv2_obj_part_pose_img = data["cv2_obj_part_pose_img"]
        meta = data["meta"]

        #####################
        # Load PoseCNN Results.
        #####################

        # gt pose.
        gt_poses = np.array(meta['poses']).flatten().reshape(3, 4, -1)

        # posecnn
        posecnn_meta_idx = str(1000000 + image_idx)[1:]  # gt results and posecnn are offset by 1
        posecnn_meta_addr = config.YCB_TOOLBOX_CONFIG + posecnn_meta_idx + config.POSECNN_EXT
        posecnn_meta = scio.loadmat(posecnn_meta_addr)

        posecnn_label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])
        poses_icp = np.array(posecnn_meta['poses_icp'])

        pred_obj_ids = np.array(posecnn_rois[:, 1], dtype=np.uint8)

        gt_obj_ids = np.array(meta['cls_indexes'].flatten(), dtype=np.uint8)
        gt_poses = np.array(meta['poses']).flatten().reshape(3, 4, -1)

        gt_to_pred_idxs = []
        for pred_obj_id in pred_obj_ids:
            if pred_obj_id in gt_obj_ids.tolist():
                gt_to_pred_idxs.append(gt_obj_ids.tolist().index(pred_obj_id))

        print("\nPred [{}]: {},\tGT [{}]: {}".format(len(pred_obj_ids), pred_obj_ids, len(gt_obj_ids), gt_obj_ids))

        #####################
        #####################

        # TODO: MATLAB EVAL
        class_ids_list = []
        pose_est_gt = []
        pose_est_df_wo_refine = []
        pose_est_df_iterative = []
        choose_list = []
        pred_c_list = []

        gt_to_pred_idx = 0
        for pred_idx, pred_obj_id in enumerate(pred_obj_ids):
            if pred_obj_id in gt_obj_ids:

                # TODO: MATLAB EVAL
                class_ids_list.append(pred_obj_id)

                obj_color = ycb_aff_dataset_utils.obj_color_map(pred_obj_id)
                print("Object: ID:{}, Name:{}".format(pred_obj_id, dataloader.obj_classes[int(pred_obj_id) - 1]))

                gt_idx = gt_to_pred_idxs[gt_to_pred_idx]
                gt_obj_id = gt_obj_ids[gt_idx]
                # print("pred\t idx:{},\t class id:{}".format(pred_idx, pred_obj_id))
                # print("gt  \t idx:{},\t class id:{}".format(gt_idx, gt_obj_id))
                gt_to_pred_idx += 1

                try:

                    #######################################
                    # bbox
                    #######################################

                    rmin, rmax, cmin, cmax = get_posecnn_bbox(posecnn_rois, pred_idx)

                    #######################################
                    # real cam for test frames
                    #######################################
                    cam_cx = config.CAM_CX_1
                    cam_cy = config.CAM_CY_1
                    cam_fx = config.CAM_FX_1
                    cam_fy = config.CAM_FY_1

                    #######################################
                    #######################################

                    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth_16bit, 0))
                    if USE_PRED_MASKS:
                        mask_label = ma.getmaskarray(ma.masked_equal(posecnn_label, pred_obj_id))
                    else:
                        mask_label = ma.getmaskarray(ma.masked_equal(obj_label, pred_obj_id))
                    mask = mask_label * mask_depth

                    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                    obj_choose = len(choose.copy())

                    if len(choose) == 0:
                        raise ZeroDivisionError
                    elif len(choose) > config.NUM_PT:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:config.NUM_PT] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, config.NUM_PT - len(choose)), 'wrap')

                    depth_masked = depth_16bit[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
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
                    img_masked = img_norm(torch.from_numpy(img_masked.astype(np.float32)))
                    index = torch.LongTensor([pred_obj_id - 1])

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

                    how_max = how_max.detach().clone().cpu().numpy()[0]

                    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                    my_pred = np.append(my_r, my_t)
                    # TODO: MATLAB EVAL
                    pose_est_df_wo_refine.append(my_pred.tolist())

                    for ite in range(0, config.REFINE_ITERATIONS):
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(config.NUM_PT,1).contiguous().view(1, config.NUM_PT, 3)
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

                    # TODO: MATLAB EVAL
                    pose_est_df_iterative.append(my_pred.tolist())
                    # choose_list.append(obj_choose)
                    # pred_c_list.append(how_max)

                    #######################################
                    # gt
                    #######################################

                    gt_pose = gt_poses[:, :, gt_idx]
                    gt_obj_r = gt_pose[0:3, 0:3]
                    gt_obj_t = gt_pose[0:3, -1]

                    gt_quart = quaternion_from_matrix(gt_obj_r)
                    my_pred = np.append(np.array(gt_quart), np.array(gt_obj_t))
                    # TODO: MATLAB EVAL
                    pose_est_gt.append(my_pred.tolist())

                    ############################
                    # Stats
                    ############################

                    stats_pred_class_ids[image_idx, pred_idx] = pred_obj_id
                    stats_pred_choose[image_idx, pred_idx] = obj_choose
                    stats_pred_c[image_idx, pred_idx] = how_max

                    #######################################
                    # Error Metrics.
                    #######################################

                    if VISUALIZE_AND_GET_ERROR_METRICS:
                        # pred
                        pred_obj_t, pred_obj_q = my_t, my_r
                        pred_obj_r = quaternion_matrix(pred_obj_q)[0:3, 0:3]
                        # eval pose.
                        eval_utils.get_error_metrics(gt_obj_t=gt_obj_t, gt_obj_r=gt_obj_r,
                                                     pred_obj_t=pred_obj_t, pred_obj_r=pred_obj_r,
                                                     refinement_idx=ite+1,
                                                     choose=obj_choose, pred_c=how_max,
                                                     verbose=True)

                    #######################################
                    # plotting pred pose.
                    #######################################

                    if VISUALIZE_AND_GET_ERROR_METRICS:
                        obj_cld = dataloader.cld[pred_obj_id]

                        # projecting 3D model to 2D image
                        imgpts, jac = cv2.projectPoints(obj_cld * 1e3, pred_obj_r, pred_obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                        if PROJECT_MESH_ON_IMAGE:
                            cv2_obj_pose_img = cv2.polylines(cv2_obj_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                        # draw pose
                        rotV, _ = cv2.Rodrigues(pred_obj_r)
                        points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                        axisPoints, _ = cv2.projectPoints(points, rotV, pred_obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)

                        axis_color = (255, 255, 255)
                        cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), axis_color, 3)
                        cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), axis_color, 3)
                        cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), axis_color, 3)

                except ZeroDivisionError:
                    print("DenseFusion Detector Lost keyframe ..")
                    # TODO: MATLAB EVAL
                    pose_est_df_wo_refine.append([0.0 for i in range(7)])
                    pose_est_df_iterative.append([0.0 for i in range(7)])

        print('Average Time for Pred: {:.3f} [s]'.format((time.time()-t0)/len(gt_obj_ids)))

        #####################
        # PLOTTING
        #####################

        if VISUALIZE_AND_GET_ERROR_METRICS:
            cv2.imshow('depth', depth_8bit)
            cv2.imshow('cv2_obj_pose_img', cv2.cvtColor(cv2_obj_pose_img, cv2.COLOR_BGR2RGB))

            cv2.waitKey(0)

        ############################
        # TODO: MATLAB EVAL
        ############################

        scio.savemat('{0}/{1}.mat'.format(config.OBJ_EVAL_FOLDER_GT, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_gt})
        scio.savemat('{0}/{1}.mat'.format(config.OBJ_EVAL_FOLDER_DF_WO_REFINE, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_df_wo_refine})
        scio.savemat('{0}/{1}.mat'.format(config.OBJ_EVAL_FOLDER_DF_ITERATIVE, '%04d' % image_idx),
                     {"class_ids": class_ids_list, 'poses': pose_est_df_iterative})

    ############################
    # Stats
    ############################

    print('\n\n\nPrinting stats ..')
    eval_utils.get_obj_stats(stats_pred_class_ids, stats_pred_choose, stats_pred_c)

if __name__ == '__main__':
    main()