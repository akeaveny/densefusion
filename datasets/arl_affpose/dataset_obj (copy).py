import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2

#######################################
#####################################

from PIL import Image
# from PIL import ImageFile
# # https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
# ImageFile.LOAD_TRUNCATED_IMAGES = True

#######################################
#######################################

from tools.ARLAffPose.utils import helper_utils

from tools.ARLAffPose import cfg as config

from tools.ARLAffPose.utils.dataset import affpose_dataset_utils
from tools.ARLAffPose.utils.pose.load_obj_ply_files import load_obj_ply_files
from tools.ARLAffPose.utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):

        ##################################
        # init path
        ##################################

        if mode == 'train':
            self.path = config.TRAIN_FILE
        elif mode == 'test':
            self.path = config.TEST_FILE
        print(self.path)

        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        ##################################
        # image list
        ##################################

        self.list = []
        self.real = []
        # self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.real.append(input_line)
            # self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        # self.len_syn = len(self.syn)

        print("Loaded: ", len(self.list))
        print("Real Images: ", len(self.real))
        # print("SYN Images: ", len(self.syn))

        ##################################
        # IMGAUG
        ##################################

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0

        self.norm = transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)

        ##################################
        # 3D models
        ##################################

        self.symmetry_obj_idx = config.SYM_OBJECTS
        self.minimum_num_pt = config.NUM_PT_MIN
        self.num_pt_mesh_small = config.NUM_PT_MESH_SMALL
        self.num_pt_mesh_large = config.NUM_PT_MESH_LARGE
        self.refine = refine

        self.cld, self.cld_obj_centered, self.cld_obj_part_centered, \
        self.obj_classes, self.obj_part_classes, \
        self.obj_ids, self.obj_part_ids = load_obj_ply_files()

        print("************** LOADED DATASET! **************")

    def __getitem__(self, index):
        # print("index:", index)
        # print("Real:", self.real[index])

        ##################################
        # init
        ##################################

        image_addr = self.real[index].rstrip()
        dataset_dir = image_addr.split('rgb/')[0]
        image_num = image_addr.split('rgb/')[-1]

        img_addr = dataset_dir + 'rgb/' + image_num + config.RGB_EXT
        depth_addr = dataset_dir + 'depth/' + image_num + config.DEPTH_EXT
        obj_label_addr = dataset_dir + 'masks_obj/' + image_num + config.OBJ_LABEL_EXT
        meta_addr = dataset_dir + 'meta/' + image_num + config.META_EXT

        img = Image.open(img_addr)
        depth = np.array(Image.open(depth_addr))
        obj_label = np.array(Image.open(obj_label_addr))
        meta = scio.loadmat(meta_addr)

        ##################################
        # IMGAUG
        ##################################

        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img)

        ##################################
        ### RESIZE & CROP
        ##################################

        img = cv2.resize(img, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_label = cv2.resize(obj_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        img = helper_utils.crop(pil_img=img, crop_size=config.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        obj_label = helper_utils.crop(pil_img=obj_label, crop_size=config.CROP_SIZE)

        ##################################
        # select random obj id
        ##################################

        label_obj_ids = np.unique(np.array(obj_label))[1:]

        obj_ids = []
        for obj_id in label_obj_ids:
            if obj_id in self.obj_ids:
                obj_ids.append(obj_id)

        while True:
            # print("Obj IDs:{}".format(len(obj_ids)))
            obj_id = obj_ids.copy()[np.random.randint(0, len(obj_ids))]
            mask_label = ma.getmaskarray(ma.masked_equal(obj_label, obj_id))
            mask_rgb = np.repeat(mask_label, 3).reshape(obj_label.shape[0], obj_label.shape[1], -1) * img
            mask_depth = mask_label * depth
            # WE NEED AT LEAST minimum_num_pt ON DEPTH IMAGE
            # print("Obj ID:{}, Depth Image Pointcloud:{}".format(obj_id, len(mask_depth.nonzero()[0])))
            if len(mask_depth.nonzero()[0]) > self.minimum_num_pt:
                break

        # print("self: ", self.obj_ids)
        # print("label: ", label_obj_ids)
        # print("obj_id: ", obj_id)

        # # todo (visualize): RGB ROIs
        # cv2_img = helper_utils.convert_16_bit_depth_to_8_bit(mask_depth.copy())
        # img_name = config.TEST_DENSEFUSION_FOLDER + 'masked_depth.png'
        # cv2.imwrite(img_name, cv2_img)
        # # todo (visualize): Depth ROIs
        # cv2_img = mask_rgb.copy()
        # img_name = config.TEST_DENSEFUSION_FOLDER + 'masked_rgb.png'
        # cv2.imwrite(img_name, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        ##################################
        # META
        ##################################

        self.xmap = config.XMAP
        self.ymap = config.YMAP

        cam_scale = config.CAMERA_SCALE # 1000 for depth [mm] to [m]
        cam_cx = meta['cam_cx'].flatten()[0] * config.X_SCALE
        cam_cy = meta['cam_cy'].flatten()[0] * config.Y_SCALE
        cam_fx = meta['cam_fx'].flatten()[0]
        cam_fy = meta['cam_fy'].flatten()[0]

        cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
        cam_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        ##################################
        # GT POSE
        ##################################

        obj_id_idx = str(1000 + obj_id)[1:]

        obj_r = meta['obj_rotation_' + np.str(obj_id_idx)]
        obj_t = meta['obj_translation_' + np.str(obj_id_idx)]    # in [m]

        # # todo (visualize): gt pose
        # cv2_img = np.array(Image.open(img_addr))
        # cv2_img = cv2.resize(cv2_img, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        # cv2_img = helper_utils.crop(pil_img=cv2_img, crop_size=config.CROP_SIZE, is_img=True)
        # imgpts, jac = cv2.projectPoints(self.cld[obj_id] * 1e3, obj_r, obj_t * 1e3, cam_mat, cam_distortion)
        # cv2_img = cv2.polylines(np.array(cv2_img), helper_utils.sort_imgpts(imgpts), True, (0, 255, 255))
        # temp_folder = config.TEST_DENSEFUSION_FOLDER + 'pose_gt.png'
        # cv2.imwrite(temp_folder, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        ##################################
        # BBOX
        ##################################

        x1, y1, x2, y2 = get_obj_bbox(obj_label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
        # print("y1, y2, x1, x2: ", y1, y2, x1, x2)

        # # todo (visualize): bbox
        # cv2_img = np.array(Image.open(img_addr))
        # cv2_img = cv2.resize(cv2_img, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        # cv2_img = helper_utils.crop(pil_img=cv2_img, crop_size=config.CROP_SIZE, is_img=True)
        # img_name = config.TEST_DENSEFUSION_FOLDER + 'bbox.png'
        # cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.imwrite(img_name, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        ##################################
        # Select Region of Interest
        ##################################

        choose = mask_depth[y1:y2, x1:x2].flatten().nonzero()[0]

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, y1:y2, x1:x2]
        depth_masked = depth[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        ######################################
        # create point cloud from depth image
        ######################################

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        translation_noise = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        if self.add_noise:
            cloud = np.add(cloud, translation_noise)

        # # todo (visualize): pointcloud_from_depth
        # cv2_img = np.array(Image.open(img_addr))
        # cv2_img = cv2.resize(cv2_img, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        # cv2_img = helper_utils.crop(pil_img=cv2_img, crop_size=config.CROP_SIZE, is_img=True)
        # imgpts, jac = cv2.projectPoints(cloud, np.eye(3), np.zeros(shape=obj_t.shape), cam_mat, cam_distortion)
        # cv2_img = cv2.polylines(np.array(cv2_img), helper_utils.sort_imgpts(imgpts), True, (0, 255, 255))
        # temp_folder = config.TEST_DENSEFUSION_FOLDER + 'pose_pointcloud_from_depth.png'
        # cv2.imwrite(temp_folder, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        ######################################
        # create target from gt pose
        ######################################

        dellist = [j for j in range(0, len(self.cld[obj_id]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj_id]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj_id]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj_id], dellist, axis=0)

        target = np.dot(model_points, obj_r.T)
        if self.add_noise:
            target = np.add(target, obj_t + translation_noise)
        else:
            target = np.add(target, obj_t)

        # # todo (visualize): gt pose from object mesh
        # cv2_img = np.array(Image.open(img_addr))
        # cv2_img = cv2.resize(cv2_img, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        # cv2_img = helper_utils.crop(pil_img=cv2_img, crop_size=config.CROP_SIZE, is_img=True)
        # imgpts, jac = cv2.projectPoints(target, np.eye(3), np.zeros(shape=obj_t.shape), cam_mat, cam_distortion)
        # cv2_img = cv2.polylines(np.array(cv2_img), helper_utils.sort_imgpts(imgpts), True, (0, 255, 255))
        # temp_folder = config.TEST_DENSEFUSION_FOLDER + 'pose_gt_target.png'
        # cv2.imwrite(temp_folder, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        ######################################
        ######################################

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([int(obj_id) - 1])

    ######################################
    ######################################

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    ######################################
    ######################################