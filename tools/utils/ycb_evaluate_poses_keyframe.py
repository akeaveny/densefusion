#! /usr/bin/env python

import argparse
import os
import sys
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math

import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
print("ROOT_DIR", ROOT_DIR)

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

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

from sklearn.neighbors import KDTree

##################################
## GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#############
#
#############
def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[roi_idx][3]) + 1
    rmax = int(posecnn_rois[roi_idx][5]) - 1
    cmin = int(posecnn_rois[roi_idx][2]) + 1
    cmax = int(posecnn_rois[roi_idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Evaluate trained model for DenseFusion')

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/YCB_Video_Dataset',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_config', required=False, default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb/dataset_config',
                    type=str,
                    metavar="")
parser.add_argument('--ycb_toolbox_config', required=False, default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/YCB_Video_toolbox/results_PoseCNN_RSS2018/',
                    type=str,
                    metavar="")
parser.add_argument('--dataset_type', required=False, default='val',
                    type=str,
                    metavar='train or val')

parser.add_argument('--ADD', required=False, default=2,
                    type=int,
                    metavar='ADD distance to evaluate')

parser.add_argument('--train_file', required=False, default='train_data_list.txt',
                    metavar="/path/to/model weights")
parser.add_argument('--val_file', required=False, default='test_data_list.txt',
                    metavar="/path/to/refine weights")

parser.add_argument('--model', required=False, default=ROOT_DIR + '/trained_models/pretrained_ycb/pose_model_26_0.012863246640872631.pth',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--refine_model', required=False, default=ROOT_DIR + '/trained_models/pretrained_ycb/pose_refine_model_69_0.009449292959118935.pth',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--classes', required=False, default='classes.txt',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--class_ids', required=False, default='class_ids.txt',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--output_result_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb',
                    type=str,
                    metavar="")
parser.add_argument('--result_wo_refine_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb/Densefusion_wo_refine_result/',
                    type=str,
                    metavar="")
parser.add_argument('--result_refine_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb/Densefusion_iterative_result/',
                    type=str,
                    metavar="")
parser.add_argument('--error_metrics_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb/Densefusion_error_metrics_result/',
                    type=str,
                    metavar="")
parser.add_argument('--error_metrics_dir1', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb/PoseCNN_error_metrics_result/',
                    type=str,
                    metavar="")
parser.add_argument('--visualize', required=False, default=False,
                    type=str,
                    metavar="Visualize Results")
parser.add_argument('--print_output', required=False, default=False,
                    type=str,
                    metavar="Visualize Results")
parser.add_argument('--save_images_path', required=False, default='/data/Akeaveny/Datasets/YCB_Video_Dataset/test_densefusion/',
                    type=str,
                    metavar="")

args = parser.parse_args()

num_obj = 21

num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1

norm_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0

img_width = 480
img_length = 640

##################################
## classes
##################################

class_file = open('{}/{}'.format(args.dataset_config, args.classes))
class_id_file = open('{}/{}'.format(args.dataset_config, args.class_ids))
classes = np.loadtxt(class_file, dtype=np.str)
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
## print("Classes: ", class_IDs)

##################################
## 3D MODELS
##################################

cld = {}
for idx, class_id in enumerate(class_IDs):
    input_file = open('{0}/models/{1}/points.xyz'.format(args.dataset, classes[idx]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

##################################
## DENSEFUSION
##################################

estimator = PoseNet(num_points=num_points, num_obj=num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(args.model))
estimator.eval()

refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(args.refine_model))
refiner.eval()

##################################
## LOAD SYN IMAGES
##################################

loaded_images_ = np.loadtxt('{}'.format('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/YCB_Video_toolbox/keyframe.txt'), dtype=np.str)

############
#
############
num_total = 0
num_correct_densefusion = 0
num_correct_pose_cnn = 0

# for image_idx in range(len(loaded_images_)):
for image_idx in range(0, 2949):

    ##############
    ##############
    str_num = loaded_images_[image_idx].split('/')[-1]

    count = 1000000 + image_idx
    meta_idx = str(count)[1:]

    rgb_addr = args.dataset + '/data/' + loaded_images_[image_idx] + "-color.png"
    depth_addr = args.dataset + '/data/' + loaded_images_[image_idx] + "-depth.png"
    gt_addr = args.dataset + '/data/' + loaded_images_[image_idx] + "-meta.mat"
    meta_addr = args.ycb_toolbox_config + meta_idx + '.mat'

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt_meta = scio.loadmat(gt_addr)
    posecnn_meta = scio.loadmat(meta_addr)

    # for plotting
    cv2_img_pose_cnn = np.array(img).copy()
    cv2_img_densefusion = np.array(img).copy()

    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    gt_poses = np.array(posecnn_meta['poses_icp'])

    lst = np.array(posecnn_rois[:, 1:2].flatten(), dtype=int)

    gt_test_pose = np.array(gt_meta['poses']).flatten().reshape(3, 4, -1)
    gt_cls_indexes = np.array(gt_meta['cls_indexes'].flatten(), dtype=int)

    roi_idxs = []
    for value in lst:
        if value in gt_cls_indexes.tolist():
            roi_idxs.append(gt_cls_indexes.tolist().index(value))

    # if args.print_output:
    print("\npred_cls_indexes: {}".format(lst))
    print("gt_cls_indexes: {}".format(gt_cls_indexes))
    print("gt_idx: {}".format(roi_idxs))

    # if args.visualize:
    #     plt.subplot(2, 2, 1)
    #     plt.title("rgb")
    #     plt.imshow(img)
    #     plt.subplot(2, 2, 2)
    #     plt.title("depth")
    #     plt.imshow(depth)
    #     plt.subplot(2, 2, 3)
    #     plt.title("gt")
    #     plt.imshow(label)
    #     plt.subplot(2, 2, 4)
    #     plt.title("label")
    #     plt.imshow(label)
    #     plt.show()
    #     plt.ioff()

    ####################
    # class ids
    ####################

    Class_id_list, ADD_list, ADD_S_list, R_list, T_list = [], [], [], [], []
    Class_id_list1, ADD_list1, ADD_S_list1, R_list1, T_list1 = [], [], [], [], []
    my_result_wo_refine = []
    my_result = []

    for class_idx in range(len(gt_cls_indexes)):
        num_total += 1

        itemid = gt_cls_indexes[class_idx]
        print("\n*** {} ***".format(classes[int(itemid) - 1]))

        try:
            if itemid not in lst:
                cld_image_gt = Image.open(rgb_addr)

                ADD = np.inf
                ADD_S = np.inf
                T_error = np.inf
                R_error = np.inf

                Class_id_list.append(itemid)
                ADD_list.append(ADD)
                ADD_S_list.append(ADD_S)
                T_list.append(T_error)
                R_list.append(R_error)

                ##############

                Class_id_list1.append(itemid)
                ADD_list1.append(ADD)
                ADD_S_list1.append(ADD_S)
                T_list1.append(T_error)
                R_list1.append(R_error)

            else:

                roi_idx = roi_idxs.index(class_idx)
                print(class_idx, roi_idx)

                rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)
                print('bbox: ', rmin, rmax, cmin, cmax)

                # img_bbox = np.array(img.copy())
                # img_bbox = cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), 255, 2)
                # cv2.imshow('img_bbox', cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
                # cv2.waitKey(0)

                # # disp
                img_bbox = np.array(img.copy())
                img_name = args.save_images_path + 'test1.bbox.png'
                cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
                cv2.imwrite(img_name, img_bbox)
                bbox_image = cv2.imread(img_name)

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm_(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                my_result_wo_refine.append(my_pred.tolist())

                for ite in range(0, iteration):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,1).contiguous().view(1, num_points, 3)
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

                ############################
                # project to screen
                ############################

                cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

                ############################
                # gt
                ############################

                gt_pose = gt_test_pose[:, :, class_idx]

                gt_rot1 = gt_pose[0:3, 0:3]
                gt_trans = gt_pose[0:3, -1]

                np.random.seed(0)
                # cld_idx = np.random.choice(np.arange(0, int(len(cld[itemid])), 1), size=int(num_points), replace=False)
                cld_idx = np.random.choice(np.arange(0, int(len(cld[itemid])), 1), size=int(len(cld[itemid])), replace=False)
                cloud = cld[itemid][cld_idx]

                imgpts_gt, jac = cv2.projectPoints(cloud, gt_rot1, gt_trans, cam_mat, dist)
                cv2_img_pose_cnn = cv2.polylines(cv2_img_pose_cnn, np.int32([np.squeeze(imgpts_gt)]), True, (0, 255, 255))
                cv2_img_densefusion = cv2.polylines(cv2_img_densefusion, np.int32([np.squeeze(imgpts_gt)]), True, (0, 255, 255))

                ############################
                # pose_cnn
                ############################

                pose_cnn_pose = gt_poses[roi_idx, :]

                pose_cnn_rot1 = quaternion_matrix(pose_cnn_pose[0:4])[0:3, 0:3]
                pose_cnn_trans = pose_cnn_pose[4:7]

                imgpts_gt, jac = cv2.projectPoints(cloud, pose_cnn_rot1, pose_cnn_trans, cam_mat, dist)
                cv2_img_pose_cnn = cv2.polylines(cv2_img_pose_cnn, np.int32([np.squeeze(imgpts_gt)]), True, (255, 0, 255))

                ############################
                # pred
                ############################

                mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                my_t = my_t

                imgpts, jac = cv2.projectPoints(cloud, mat_r, my_t, cam_mat, dist)
                cv2_img_densefusion = cv2.polylines(cv2_img_densefusion, np.int32([np.squeeze(imgpts)]), True, (255, 255, 0))

                ############################
                # POSECNN: Error Metrics
                ############################

                T_pred, T_gt = pose_cnn_trans, gt_trans
                R_pred, R_gt = pose_cnn_rot1, gt_rot1

                # ADD
                pred = np.dot(cld[itemid], R_pred)
                pred = np.add(pred, T_pred)

                target = np.dot(cld[itemid], R_gt)
                target = np.add(target, T_gt)

                ADD = np.mean(np.linalg.norm(pred - target, axis=1))

                # ADD-S
                tree = KDTree(pred)
                dist, ind = tree.query(target)
                ADD_S = np.mean(dist)

                # translation
                T_error = np.linalg.norm(T_pred - T_gt)

                # rot
                error_cos = 0.5 * (np.trace(R_pred @ np.linalg.inv(R_gt)) - 1.0)
                error_cos = min(1.0, max(-1.0, error_cos))
                error = np.arccos(error_cos)
                R_error = 180.0 * error / np.pi

                print("PoseCNN:")
                print("\tADD: {:.2f} [cm]".format(ADD * 100))  # [cm]
                print("\tADD-S: {:.2f} [cm]".format(ADD_S * 100))
                print("\tT: {:.2f} [cm]".format(T_error * 100))  # [cm]
                print("\tRot: {:.2f} [deg]".format(R_error))

                Class_id_list1.append(itemid)
                ADD_list1.append(ADD)
                ADD_S_list1.append(ADD_S)
                T_list1.append(T_error)
                R_list1.append(R_error)

                ADD = min(ADD, ADD_S) * 100  # [cm] # TODO: ADD-S from cls list
                if ADD < args.ADD:
                    num_correct_pose_cnn += 1
                    print('{} Pass! ADD: {:.2f}'.format(classes[int(itemid) - 1], ADD))
                else:
                    print('{} NOT Pass! ADD: {:.2f}'.format(classes[int(itemid) - 1], ADD))
                print('*** Num Correct:{}/{}.. ***'.format(num_correct_pose_cnn, num_total))

                ############################
                # DENSEFUSION: Error Metrics
                ############################

                T_pred, T_gt = my_t, gt_trans
                R_pred, R_gt = mat_r, gt_rot1

                # ADD
                pred = np.dot(cld[itemid], R_pred)
                pred = np.add(pred,  T_pred)

                target = np.dot(cld[itemid], R_gt)
                target = np.add(target, T_gt)

                ADD = np.mean(np.linalg.norm(pred - target, axis=1))

                # ADD-S
                tree = KDTree(pred)
                dist, ind = tree.query(target)
                ADD_S = np.mean(dist)

                # translation
                T_error = np.linalg.norm(T_pred - T_gt)

                # rot
                error_cos = 0.5 * (np.trace( R_pred @ np.linalg.inv(R_gt)) - 1.0)
                error_cos = min(1.0, max(-1.0, error_cos))
                error = np.arccos(error_cos)
                R_error = 180.0 * error / np.pi

                print("DenseFusion:")
                print("\tADD: {:.2f} [cm]".format(ADD * 100))  # [cm]
                print("\tADD-S: {:.2f} [cm]".format(ADD_S * 100))
                print("\tT: {:.2f} [cm]".format(T_error * 100))  # [cm]
                print("\tRot: {:.2f} [deg]".format(R_error))

                Class_id_list.append(itemid)
                ADD_list.append(ADD)
                ADD_S_list.append(ADD_S)
                T_list.append(T_error)
                R_list.append(R_error)

                ADD = min(ADD, ADD_S) * 100 # [cm] # TODO: ADD-S from cls list
                if ADD < args.ADD:
                    num_correct_densefusion += 1
                    print('{} Pass! ADD: {:.2f}'.format(classes[int(itemid) - 1], ADD))
                else:
                    print('{} NOT Pass! ADD: {:.2f}'.format(classes[int(itemid) - 1], ADD))
                print('*** Num Correct:{}/{}.. ***'.format(num_correct_densefusion, num_total))

                ############################
                ############################
                # cv2.imshow("PoseCNN: ", cv2.cvtColor(cv2_img_pose_cnn, cv2.COLOR_RGB2BGR))
                # cv2.imshow("DenseFusion: ", cv2.cvtColor(cv2_img_densefusion, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

        except ZeroDivisionError:
            print("DenseFusion Detector Lost {0} at No.{1} keyframe".format(itemid, class_idx))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

    ############################
    ############################
    cv2.imshow("PoseCNN: ", cv2.cvtColor(cv2_img_pose_cnn, cv2.COLOR_RGB2BGR))
    cv2.imshow("DenseFusion: ", cv2.cvtColor(cv2_img_densefusion, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    scio.savemat('{0}/{1}.mat'.format(args.result_wo_refine_dir, '%04d' % image_idx), {'poses': my_result_wo_refine})
    scio.savemat('{0}/{1}.mat'.format(args.result_refine_dir, '%04d' % image_idx), {'poses': my_result})
    scio.savemat('{0}/{1}.mat'.format(args.error_metrics_dir, '%04d' % image_idx),
                     {"Class_IDs": Class_id_list, "ADD": ADD_list, "ADD_S": ADD_S_list, "R": R_list, "T": T_list})
    scio.savemat('{0}/{1}.mat'.format(args.error_metrics_dir1, '%04d' % image_idx),
                 {"Class_IDs": Class_id_list1, "ADD": ADD_list1, "ADD_S": ADD_S_list1, "R": R_list1, "T": T_list1})
    print("******************* Finish No.{0} keyframe *******************\n".format(image_idx))