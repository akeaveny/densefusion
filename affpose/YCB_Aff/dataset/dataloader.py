import os
import glob
import json

import numpy as np

import cv2
from PIL import Image

import scipy.io as scio

#######################################
#######################################

import sys
sys.path.append('../../..')

#######################################
#######################################

from affpose.YCB_Aff.utils import helper_utils

from affpose.YCB_Aff import cfg as config
from affpose.YCB_Aff.dataset import ycb_aff_dataset_utils

from affpose.YCB_Aff.utils.pose.load_obj_part_ply_files import load_obj_part_ply_files
from affpose.YCB_Aff.utils.bbox.extract_bboxs_from_label import get_obj_part_bbox

#######################################
#######################################

class YCBAff():

    def __init__(self,
                 split='train',
                 use_pred_masks=False,
                 select_random_images=False,
                 num_images=100,
                 ):

        self.use_pred_masks = use_pred_masks

        ###################################
        # Load Ply files
        ###################################

        self.cld, self.cld_obj_centered, self.cld_obj_part_centered, \
        self.obj_classes, self.obj_part_classes, \
        self.obj_ids, self.obj_part_ids = load_obj_part_ply_files()

        ##################################
        # Load Images
        ##################################

        self.split = split
        assert self.split == 'train' or self.split == 'val' or self.split == 'test'

        if self.split == 'train':
            image_files = open('{}'.format(config.TRAIN_FILE), "r")
        elif self.split == 'test':
            image_files = open('{}'.format(config.TEST_FILE), "r")
        self.img_files = np.sort(np.array(image_files.readlines()))
        print("Loaded Files: {}".format(len(self.img_files)))

        total_idx = np.arange(0, len(self.img_files), 25)
        self.img_files = self.img_files[total_idx]
        print("Chosen Files: {}".format(len(self.img_files)))

        if select_random_images:
            np.random.seed(0)
            idx = np.arange(0, len(self.img_files), 1)
            random_idx = np.random.choice(idx, size=int(num_images), replace=False)
            self.img_files = np.array(self.img_files)[random_idx]
            print("Chosen Files: {}".format(len(self.img_files)))

    def get_item(self, image_idx, verbose=False):

        image_addr = self.img_files[image_idx].rstrip()
        image_addr = image_addr.rstrip()

        print('\nimage:{}/{}, file:{}'.format(image_idx + 1, len(self.img_files), image_addr))

        rgb_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.RGB_EXT
        depth_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.DEPTH_EXT
        obj_label_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.LABEL_EXT
        obj_part_label_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.OBJ_PART_LABEL_EXT
        aff_label_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.AFF_LABEL_EXT
        meta_addr = config.AFF_DATASET_ROOT_PATH + image_addr + config.META_EXT

        rgb = np.array(Image.open(rgb_addr))[..., :3]
        depth = np.array(Image.open(depth_addr))
        obj_label = np.array(Image.open(obj_label_addr))
        obj_part_label = np.array(Image.open(obj_part_label_addr))
        aff_label = np.array(Image.open(aff_label_addr))
        meta = scio.loadmat(meta_addr)

        ##################################
        # TODO: Load PRED masks
        ##################################

        # if self.use_pred_masks:
        #     obj_label_addr = dataset_dir + 'pred_obj/' + image_num + config.TEST_OBJ_PRED_EXT
        #     obj_part_label_addr = dataset_dir + 'pred_aff/' + image_num + config.TEST_OBJ_PART_PRED_EXT
        #     aff_label_addr = dataset_dir + 'pred_aff/' + image_num + config.TEST_OBJ_PRED_EXT
        #
        #     obj_label = np.array(Image.open(obj_label_addr))
        #     obj_part_label = np.array(Image.open(obj_part_label_addr))
        #     aff_label = np.array(Image.open(aff_label_addr))

        ##################################
        # CAMERA
        ##################################

        self.cam_scale = meta['factor_depth'][0][0]
        self.cam_cx = meta['cam_cx'].flatten()[0]
        self.cam_cy = meta['cam_cy'].flatten()[0]
        self.cam_fx = meta['cam_fx'].flatten()[0]
        self.cam_fy = meta['cam_fy'].flatten()[0]

        self.cam_mat = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]])
        self.cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        #####################
        # Image utils
        #####################

        # OBJ
        colour_obj_label = ycb_aff_dataset_utils.colorize_obj_mask(obj_label)
        colour_obj_label = cv2.addWeighted(rgb, 0.4, colour_obj_label, 0.6, 0)

        # Img to draw 6-DoF Pose.
        cv2_obj_pose_img = colour_obj_label.copy()

        # AFF
        colour_aff_label = ycb_aff_dataset_utils.colorize_aff_mask(aff_label)
        colour_aff_label = cv2.addWeighted(rgb, 0.4, colour_aff_label, 0.6, 0)

        # OBJ PART
        # obj_label = ycb_aff_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_label)
        # colour_obj_part_label = ycb_aff_dataset_utils.colorize_obj_mask(obj_label)
        # colour_obj_part_label = cv2.addWeighted(rgb, 0.35, colour_obj_part_label, 0.65, 0)

        # Img to draw 6-DoF Pose.
        cv2_obj_part_pose_img = colour_aff_label.copy()

        #####################
        #####################

        return {"rgb": rgb,
                "depth_16bit": depth,
                "depth_8bit": helper_utils.convert_16_bit_depth_to_8_bit(depth),
                "obj_label": obj_label,
                "obj_part_label": obj_part_label,
                "aff_label": aff_label,
                "cv2_obj_pose_img": cv2_obj_pose_img,
                "cv2_obj_part_pose_img": cv2_obj_part_pose_img,
                "meta": meta,
                }
    
    def draw_gt_obj_pose(self, image_idx, verbose=False, project_mesh_on_image=False, get_occlusion_metrics=False):

        data = self.get_item(image_idx)

        rgb = data["rgb"]
        depth_16bit = data["depth_16bit"]
        depth_8bit = data["depth_8bit"]
        obj_label = data["obj_label"]
        obj_part_label = data["obj_part_label"]
        cv2_obj_pose_img = data["cv2_obj_pose_img"]
        cv2_obj_part_pose_img = data["cv2_obj_part_pose_img"]
        meta = data["meta"]

        obj_part_occlusion_mask = {}

        #######################################
        # OBJECT
        #######################################

        # obj ids.
        obj_ids = np.array(meta['cls_indexes']).flatten()
        label_obj_ids = np.unique(obj_label)[1:]
        label_obj_part_ids = np.unique(obj_part_label)[1:]
        if verbose:
            print('GT obj id: {},\nPred obj id: {},\tPred obj part ids: {}'
                  .format(obj_ids, label_obj_ids, label_obj_part_ids))

        # gt pose.
        gt_poses = np.array(meta['poses']).flatten().reshape(3, 4, -1)

        for idx, obj_id in enumerate(obj_ids):
            if obj_id in label_obj_ids:
                obj_id = int(obj_id)
                obj_name = "{:<15}".format(ycb_aff_dataset_utils.map_obj_id_to_name(obj_id))
                obj_color = ycb_aff_dataset_utils.obj_color_map(obj_id)
                if verbose:
                    print("\tObject: {} Id: {}".format(obj_name, obj_id))

                obj_meta_idx = str(1000 + obj_id)[1:]
                obj_r = gt_poses[:, :, idx][0:3, 0:3]
                obj_t = gt_poses[:, :, idx][0:3, -1]

                obj_r = np.array(obj_r, dtype=np.float64).reshape(3, 3)
                obj_t = np.array(obj_t, dtype=np.float64).reshape(-1, 3)

                ##################################
                # BBOX
                ##################################

                x1, y1, x2, y2 = get_obj_part_bbox(obj_label.copy(), obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                # cv2_obj_pose_img = cv2.rectangle(cv2_obj_pose_img, (x1, y1), (x2, y2), obj_color, 2)
                # depth_8bit = cv2.rectangle(depth_8bit, (x1, y1), (x2, y2), 255, 2)

                #######################################
                # ITERATE OVER OBJ PARTS
                #######################################

                obj_part_ids = ycb_aff_dataset_utils.map_obj_ids_to_obj_part_ids(obj_id)
                if verbose:
                    print('\tobj_part_ids:{}'.format(obj_part_ids))
                for obj_part_id in obj_part_ids:
                    if obj_part_id in label_obj_part_ids:
                        obj_part_id = int(obj_part_id)
                        aff_id = ycb_aff_dataset_utils.map_obj_part_ids_to_aff_ids(obj_part_id)
                        if verbose:
                            print("\t\tObj Part Id: {}".format(obj_part_id))

                        #######################################
                        # OBJECT POSE
                        #######################################

                        if project_mesh_on_image:

                            # projecting 3D model to 2D image
                            obj_centered = self.cld_obj_centered[obj_part_id]
                            imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, self.cam_mat, self.cam_dist)
                            cv2_obj_pose_img = cv2.polylines(cv2_obj_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                            # draw pose
                            rotV, _ = cv2.Rodrigues(obj_r)
                            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                            axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, self.cam_mat, self.cam_dist)
                            cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                            cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                            cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                        #######################################
                        # OBJECT PART AFF CENTERED
                        #######################################
                        aff_color = ycb_aff_dataset_utils.aff_color_map(aff_id)

                        obj_part_centered = self.cld_obj_part_centered[obj_part_id]

                        obj_part_id_idx = str(1000 + obj_part_id)[1:]
                        obj_part_r = meta['obj_part_rotation_' + np.str(obj_part_id_idx)]
                        obj_part_t = meta['obj_part_translation_' + np.str(obj_part_id_idx)]

                        #######################################
                        # BBOX
                        #######################################

                        if obj_part_id in self.obj_part_ids:
                            obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_part_bbox(obj_part_label.copy(), obj_part_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                            # cv2_obj_part_pose_img = cv2.rectangle(cv2_obj_part_pose_img, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), aff_color, 2)
                            # depth_8bit = cv2.rectangle(depth_8bit, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), 128, 2)

                        ######################################
                        # 6-DOF POSE
                        ######################################

                        if project_mesh_on_image:

                            # draw model
                            obj_parts_imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, self.cam_mat, self.cam_dist)
                            cv2_obj_part_pose_img = cv2.polylines(cv2_obj_part_pose_img, np.int32([np.squeeze(obj_parts_imgpts)]), False, aff_color)

                            if obj_part_id in self.obj_part_ids:
                                # draw pose
                                rotV, _ = cv2.Rodrigues(obj_part_r)
                                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                                axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, self.cam_mat, self.cam_dist)
                                cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                                cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                                cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                        #######################################
                        # Occlusion
                        #######################################

                        if get_occlusion_metrics:
                            # get expected mask.
                            imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, self.cam_mat, self.cam_dist)
                            expected_obj_part_mask = np.zeros(shape=(config.WIDTH, config.HEIGHT), dtype=np.uint8)
                            expected_obj_part_mask = cv2.polylines(expected_obj_part_mask, np.int32([np.squeeze(imgpts)]), isClosed=False, color=1)

                            # filter out extra points drawn using cv2.polylines()
                            masked_obj_label = np.ma.getmaskarray(np.ma.masked_not_equal(obj_label, 0)).astype(np.uint8)
                            expected_obj_part_mask = cv2.bitwise_and(masked_obj_label, expected_obj_part_mask)
                            obj_part_occlusion_mask[obj_part_id] = expected_obj_part_mask

        #####################
        #####################

        data["meta"] = meta
        data["depth_8bit"] = depth_8bit
        data["cv2_obj_pose_img"] = cv2_obj_pose_img
        data["cv2_obj_part_pose_img"] = cv2_obj_part_pose_img
        data["obj_part_occlusion_mask"] = obj_part_occlusion_mask

        return data
