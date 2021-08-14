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

from affpose.ARLAffPose import cfg as config
from affpose.ARLAffPose.dataset import arl_affpose_dataset_utils
from affpose.ARLAffPose.dataset import dataloader as arl_affpose_dataloader
from affpose.ARLAffPose.utils.bbox.extract_bboxs_from_label import get_obj_bbox
from affpose.ARLAffPose.eval import eval_utils

#######################################
#######################################

DELETE_OLD_RESULTS = False
SELECT_RANDOM_IMAGES = False
NUM_IMAGES = 100
VISUALIZE_AND_GET_ERROR_METRICS = False
PROJECT_MESH_ON_IMAGE = False


def main():

    ##################################
    # Remove old results.
    ##################################

    if DELETE_OLD_RESULTS:
        files = glob.glob(config.AFF_EVAL_FOLDER_GT + '/*')
        for file in files:
            os.remove(file)

        files = glob.glob(config.AFF_EVAL_FOLDER_DF_WO_REFINE + '/*')
        for file in files:
            os.remove(file)

        files = glob.glob(config.AFF_EVAL_FOLDER_DF_ITERATIVE + '/*')
        for file in files:
            os.remove(file)

    ##################################
    # DENSEFUSION
    ##################################

    estimator = PoseNet(num_points=config.NUM_PT, num_obj=config.NUM_OBJECTS_PARTS)
    estimator.cuda()
    estimator.load_state_dict(torch.load(config.PRE_TRAINED_AFF_MODEL))
    estimator.eval()

    refiner = PoseRefineNet(num_points=config.NUM_PT, num_obj=config.NUM_OBJECTS_PARTS)
    refiner.cuda()
    refiner.load_state_dict(torch.load(config.PRE_TRAINED_AFF_REFINE_MODEL))
    refiner.eval()

    img_norm = transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)

    ###################################
    # Load ARL AFFPose
    ###################################

    # load real images.
    dataloader = arl_affpose_dataloader.ARLAffPose(split='test', select_random_images=SELECT_RANDOM_IMAGES, num_images=NUM_IMAGES)

    ###################################
    # Stats
    ###################################

    stats_pred_class_ids = np.zeros(shape=(len(dataloader.img_files), 20))
    stats_pred_occlusion = np.zeros(shape=(len(dataloader.img_files), 20))
    stats_pred_choose = np.zeros(shape=(len(dataloader.img_files), 20))
    stats_pred_c = np.zeros(shape=(len(dataloader.img_files), 20))

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
        #####################

        # TODO: MATLAB EVAL
        class_ids_list = []
        pose_est_gt = []
        pose_est_df_wo_refine = []
        pose_est_df_iterative = []
        occlusion_list = []
        choose_list = []
        pred_c_list = []

        #####################
        #####################

        obj_ids = np.array(meta['object_class_ids']).flatten()
        for idx, obj_id in enumerate(obj_ids):
            if obj_id in np.unique(obj_label):
                obj_color = arl_affpose_dataset_utils.obj_color_map(obj_id)
                print("Object: ID:{}, Name:{}".format(obj_id, dataloader.obj_classes[int(obj_id) - 1]))

                #######################################
                # ITERATE OVER OBJ PARTS
                #######################################

                obj_part_ids = arl_affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
                for obj_part_id in obj_part_ids:
                    if obj_part_id in np.unique(obj_part_label):
                        obj_part_id = int(obj_part_id)
                        aff_id = arl_affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                        print("Object Part ID: {}, Namw: {}".format(obj_part_id, dataloader.obj_part_classes[int(obj_part_id) - 1]))

                        #######################################
                        # ground truth.
                        #######################################

                        obj_part_id_idx = str(1000 + obj_part_id)[1:]
                        gt_obj_r = meta['obj_part_rotation_' + np.str(obj_part_id_idx)]
                        gt_obj_t = meta['obj_part_translation_' + np.str(obj_part_id_idx)]
                        obj_occlusion = meta['obj_part_occlusion' + str(obj_part_id_idx)]

                        # TODO: MATLAB EVAL
                        class_ids_list.append(obj_part_id)
                        occlusion_list.append(obj_occlusion)

                        #####################
                        #####################

                        # get bbox.
                        x1, y1, x2, y2 = get_obj_bbox(obj_part_label.copy(), obj_part_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                        # get mask.
                        mask_label = ma.getmaskarray(ma.masked_equal(obj_part_label, obj_part_id))
                        mask_depth = mask_label * depth_16bit

                        try:

                            ##################################
                            # Select Region of Interest
                            ##################################

                            choose = mask_depth[y1:y2, x1:x2].flatten().nonzero()[0]
                            obj_choose = len(choose.copy())

                            if len(choose) > config.NUM_PT:
                                c_mask = np.zeros(len(choose), dtype=int)
                                c_mask[:config.NUM_PT] = 1
                                np.random.shuffle(c_mask)
                                choose = choose[c_mask.nonzero()]
                            else:
                                choose = np.pad(choose, (0, config.NUM_PT - len(choose)), 'wrap')

                            img_masked = np.transpose(np.array(rgb)[:, :, :3], (2, 0, 1))[:, y1:y2, x1:x2]
                            depth_masked = depth_16bit[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                            xmap_masked = config.XMAP[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                            ymap_masked = config.YMAP[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                            choose = np.array([choose])

                            ######################################
                            # create point cloud from depth image
                            ######################################

                            pt2 = depth_masked / config.CAMERA_SCALE
                            pt0 = (ymap_masked - dataloader.cam_cx) * pt2 / dataloader.cam_fx
                            pt1 = (xmap_masked - dataloader.cam_cy) * pt2 / dataloader.cam_fy
                            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                            ######################################
                            # Send to Torch.
                            ######################################

                            cloud = torch.from_numpy(cloud.astype(np.float32))
                            choose = torch.LongTensor(choose.astype(np.int32))
                            img_masked = img_norm(torch.from_numpy(img_masked.astype(np.float32)))
                            index = torch.LongTensor([obj_part_id - 1])  # TODO: obj part or obj_part_id

                            cloud = Variable(cloud).cuda()
                            choose = Variable(choose).cuda()
                            img_masked = Variable(img_masked).cuda()
                            index = Variable(index).cuda()

                            cloud = cloud.view(1, config.NUM_PT, 3)
                            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                            #######################################
                            # Estimate Pose.
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
                                                             refinement_idx=0,
                                                             occlusion=obj_occlusion, choose=obj_choose, pred_c=how_max,
                                                             verbose=True)

                            # TODO: MATLAB EVAL
                            if how_max > config.PRED_C_THRESHOLD:
                                pose_est_gt.append(my_pred.tolist())
                                pose_est_df_wo_refine.append(my_pred.tolist())
                                choose_list.append(obj_choose)
                                pred_c_list.append(how_max)

                            #######################################
                            # Refine Pose.
                            #######################################

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
                                                                 occlusion=obj_occlusion, choose=obj_choose, pred_c=how_max,
                                                                 verbose=True)

                            # TODO: MATLAB EVAL
                            if how_max > config.PRED_C_THRESHOLD:
                                pose_est_df_iterative.append(my_pred.tolist())

                            #######################################
                            # plotting pred pose.
                            #######################################

                            if VISUALIZE_AND_GET_ERROR_METRICS:
                                if obj_part_id in arl_affpose_dataset_utils.DRAW_OBJ_PART_POSE:
                                    obj_cld = dataloader.cld_obj_part_centered[obj_part_id]

                                    # projecting 3D model to 2D image
                                    imgpts, jac = cv2.projectPoints(obj_cld * 1e3, pred_obj_r, pred_obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                                    if PROJECT_MESH_ON_IMAGE:
                                        cv2_obj_part_pose_img = cv2.polylines(cv2_obj_part_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                                    # modify YCB objects rotation matrix
                                    _pred_obj_r = arl_affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, pred_obj_r.copy())

                                    # draw pose
                                    rotV, _ = cv2.Rodrigues(_pred_obj_r)
                                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                                    axisPoints, _ = cv2.projectPoints(points, rotV, pred_obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)

                                    axis_color = (255, 255, 255)
                                    cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), axis_color, 3)
                                    cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), axis_color, 3)
                                    cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), axis_color, 3)

                        except ZeroDivisionError:
                            print("DenseFusion Detector Lost keyframe ..")
                            # TODO: MATLAB EVAL
                            pose_est_df_wo_refine.append([0.0 for i in range(7)])
                            pose_est_df_iterative.append([0.0 for i in range(7)])
                            choose_list.append(0)
                            pred_c_list.append(0)

        print('Average Time for Pred: {:.3f} [s]'.format((time.time()-t0)/len(obj_ids)))

        #####################
        # PLOTTING
        #####################

        if VISUALIZE_AND_GET_ERROR_METRICS:
            cv2.imshow('depth', depth_8bit)
            cv2.imshow('cv2_obj_part_pose_img', cv2.cvtColor(cv2_obj_part_pose_img, cv2.COLOR_BGR2RGB))

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

        for idx in range(len(class_ids_list)):
            stats_pred_class_ids[image_idx, idx] = class_ids_list[idx]
            stats_pred_occlusion[image_idx, idx] = occlusion_list[idx]
            stats_pred_choose[image_idx, idx] = choose_list[idx]
            stats_pred_c[image_idx, idx] = pred_c_list[idx]

    print('\nPrinting stats ..')
    eval_utils.get_pbj_part_stats(stats_pred_class_ids, stats_pred_occlusion, stats_pred_choose, stats_pred_c)

if __name__ == '__main__':
    main()