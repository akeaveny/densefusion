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
from affpose.YCB_Aff.utils.bbox.extract_bboxs_from_label import get_obj_part_bbox
from affpose.YCB_Aff.eval import eval_utils

#######################################
#######################################

DELETE_OLD_RESULTS = True

SELECT_RANDOM_IMAGES = False
NUM_IMAGES = 10

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
    dataloader = ycb_aff_dataloader.YCBAff(split='test', select_random_images=SELECT_RANDOM_IMAGES, num_images=NUM_IMAGES)

    ###################################
    # Stats
    ###################################

    stats_pred_class_ids = np.zeros(shape=(len(dataloader.img_files), 10))
    stats_pred_occlusion = np.zeros(shape=(len(dataloader.img_files), 10))
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
        occlusion_list = []

        #####################
        #####################

        gt_to_pred_idx = 0
        for idx, pred_obj_id in enumerate(pred_obj_ids):
            if pred_obj_id in gt_obj_ids:

                # TODO: MATLAB EVAL
                class_ids_list.append(pred_obj_id)

                #######################################
                # GT OBJ ID
                #######################################

                idx = gt_to_pred_idxs[gt_to_pred_idx]
                gt_obj_id = gt_obj_ids[idx]
                gt_to_pred_idx += 1

                obj_color = ycb_aff_dataset_utils.obj_color_map(pred_obj_id)
                print("Object: ID:{}, Name:{}".format(pred_obj_id, dataloader.obj_classes[int(pred_obj_id) - 1]))

                obj_part_ids = ycb_aff_dataset_utils.map_obj_ids_to_obj_part_ids(pred_obj_id)
                for obj_part_id in obj_part_ids:
                    if obj_part_id in dataloader.obj_part_ids and obj_part_id in np.unique(obj_part_label):

                        #######################################
                        # ground truth.
                        #######################################

                        obj_id_idx = str(1000 + gt_obj_id)[1:]
                        obj_occlusion = meta['obj_occlusion' + str(obj_id_idx)]

                        gt_obj_r = gt_poses[:, :, idx][0:3, 0:3]
                        gt_obj_t = gt_poses[:, :, idx][0:3, -1]

                        gt_obj_q = quaternion_from_matrix(gt_obj_r)
                        gt_obj_pose = np.append(np.array(gt_obj_q), np.array(gt_obj_t))
                        pose_est_gt.append(gt_obj_pose.tolist())

                        try:

                            ##################################
                            # OBJECT: Select Region of Interest
                            ##################################
                            # get bbox.
                            x1, y1, x2, y2 = get_obj_part_bbox(obj_label.copy(), pred_obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                            # get mask.
                            mask_label = ma.getmaskarray(ma.masked_equal(obj_label, pred_obj_id))
                            mask_depth = mask_label * depth_16bit

                            choose = mask_depth[y1:y2, x1:x2].flatten().nonzero()[0]
                            obj_choose = len(choose.copy())
                            
                            ##################################
                            # OBJECT PART: Select Region of Interest
                            ##################################
                            
                            # get bbox.
                            obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_part_bbox(obj_part_label.copy(), obj_part_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                            # get mask.
                            obj_part_mask_label = ma.getmaskarray(ma.masked_equal(obj_part_label, obj_part_id))
                            obj_part_mask_depth = obj_part_mask_label * depth_16bit

                            obj_part_choose = obj_part_mask_depth[obj_part_y1:obj_part_y2, obj_part_x1:obj_part_x2].flatten().nonzero()[0]
                            obj_part_choose = len(obj_part_choose.copy())

                            ##################################
                            ##################################

                            if obj_part_choose == 0:
                                raise ZeroDivisionError
                            elif len(choose) > config.NUM_PT:
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

                            pt2 = depth_masked / dataloader.cam_scale
                            pt0 = (ymap_masked - dataloader.cam_cx) * pt2 / dataloader.cam_fx
                            pt1 = (xmap_masked - dataloader.cam_cy) * pt2 / dataloader.cam_fy
                            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                            ######################################
                            # Send to Torch.
                            ######################################

                            cloud = torch.from_numpy(cloud.astype(np.float32))
                            choose = torch.LongTensor(choose.astype(np.int32))
                            img_masked = img_norm(torch.from_numpy(img_masked.astype(np.float32)))
                            index = torch.LongTensor([pred_obj_id - 1])  # TODO: obj part or obj_part_id

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
                                pose_est_df_wo_refine.append(my_pred.tolist())
                                choose_list.append(obj_choose)
                                pred_c_list.append(how_max)
                                occlusion_list.append(obj_occlusion)

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
                            choose_list.append(0)
                            pred_c_list.append(0)
                            occlusion_list.append(0)

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

        for idx in range(len(class_ids_list)):
            stats_pred_class_ids[image_idx, idx] = class_ids_list[idx]
            stats_pred_occlusion[image_idx, idx] = occlusion_list[idx]
            stats_pred_choose[image_idx, idx] = choose_list[idx]
            stats_pred_c[image_idx, idx] = pred_c_list[idx]

    print('\nPrinting stats ..')
    # eval_utils.get_obj_stats(stats_pred_class_ids, stats_pred_occlusion, stats_pred_choose, stats_pred_c)

if __name__ == '__main__':
    main()