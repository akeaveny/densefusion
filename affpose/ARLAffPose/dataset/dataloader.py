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

from affpose.ARLAffPose.utils import helper_utils

from affpose.ARLAffPose import cfg as config
from affpose.ARLAffPose.dataset import arl_affpose_dataset_utils

from affpose.ARLAffPose.utils.pose.load_obj_ply_files import load_obj_ply_files
from affpose.ARLAffPose.utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

class ARLAffPose():

    def __init__(self,
                 split='train',
                 select_random_images=True,
                 num_images=250,
                 ):

        ###################################
        # Load Ply files
        ###################################

        self.cld, self.cld_obj_centered, self.cld_obj_part_centered, \
        self.obj_classes, self.obj_part_classes, \
        self.obj_ids, self.obj_part_ids = load_obj_ply_files()

        ##################################
        # Load Images
        ##################################

        self.split = split
        assert self.split == 'train' or self.split == 'val' or self.split == 'test'

        if self.split == 'train':
            image_files = open('{}'.format(config.TRAIN_FILE), "r")
        elif self.split == 'val':
            image_files = open('{}'.format(config.VAL_FILE), "r")
        elif self.split == 'test':
            image_files = open('{}'.format(config.TEST_FILE), "r")
        self.img_files = np.sort(np.array(image_files.readlines()))
        print("Loaded Files: {}".format(len(self.img_files)))

        if select_random_images:
            np.random.seed(0)
            idx = np.arange(0, len(self.img_files), 1)
            random_idx = np.random.choice(idx, size=int(num_images), replace=False)
            self.img_files = np.array(self.img_files)[random_idx]
            print("Chosen Files: {}".format(len(self.img_files)))

    def get_item(self, image_idx):

        image_addr = self.img_files[image_idx].rstrip()
        dataset_dir = image_addr.split('rgb/')[0]
        image_num = image_addr.split('rgb/')[-1]

        print('\nimage:{}/{}, file:{}'.format(image_idx + 1, len(self.img_files), image_addr))

        rgb_addr = dataset_dir + 'rgb/' + image_num + config.RGB_EXT
        depth_addr = dataset_dir + 'depth/' + image_num + config.DEPTH_EXT
        obj_label_addr = dataset_dir + 'masks_obj/' + image_num + config.OBJ_LABEL_EXT
        obj_part_label_addr = dataset_dir + 'masks_obj_part/' + image_num + config.OBJ_PART_LABEL_EXT
        aff_label_addr = dataset_dir + 'masks_aff/' + image_num + config.AFF_LABEL_EXT
        meta_addr = dataset_dir + 'meta/' + image_num + config.META_EXT

        rgb = np.array(Image.open(rgb_addr))[..., :3]
        depth = np.array(Image.open(depth_addr))
        obj_label = np.array(Image.open(obj_label_addr))
        obj_part_label = np.array(Image.open(obj_part_label_addr))
        aff_label = np.array(Image.open(aff_label_addr))
        meta = scio.loadmat(meta_addr)

        ##################################
        # RESIZE & CROP
        ##################################

        rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        obj_label = cv2.resize(obj_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_part_label = cv2.resize(obj_part_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        aff_label = cv2.resize(aff_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        rgb = helper_utils.crop(pil_img=rgb, crop_size=config.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        obj_label = helper_utils.crop(pil_img=obj_label, crop_size=config.CROP_SIZE)
        obj_part_label = helper_utils.crop(pil_img=obj_part_label, crop_size=config.CROP_SIZE)
        aff_label = helper_utils.crop(pil_img=aff_label, crop_size=config.CROP_SIZE)

        ##################################
        # Load PRED masks
        ##################################

        if self.split == 'test':
            obj_label_addr = dataset_dir + 'pred_obj/' + image_num + config.TEST_OBJ_PRED_EXT
            obj_part_label_addr = dataset_dir + 'pred_aff/' + image_num + config.TEST_OBJ_PART_PRED_EXT
            aff_label_addr = dataset_dir + 'pred_aff/' + image_num + config.TEST_OBJ_PRED_EXT

            obj_label = np.array(Image.open(obj_label_addr))
            obj_part_label = np.array(Image.open(obj_part_label_addr))
            aff_label = np.array(Image.open(aff_label_addr))

        ##################################
        # CAMERA
        ##################################

        self.cam_cx = meta['cam_cx'].flatten()[0]
        self.cam_cy = meta['cam_cy'].flatten()[0]
        self.cam_fx = meta['cam_fx'].flatten()[0]
        self.cam_fy = meta['cam_fy'].flatten()[0]

        self.og_cam_mat = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]])
        self.og_cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.cam_cx = meta['cam_cx'].flatten()[0] * config.X_SCALE
        self.cam_cy = meta['cam_cy'].flatten()[0] * config.Y_SCALE
        self.cam_fx = meta['cam_fx'].flatten()[0]
        self.cam_fy = meta['cam_fy'].flatten()[0]

        self.cam_mat = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]])
        self.cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        #####################
        # Image utils
        #####################

        # OBJ
        colour_obj_label = arl_affpose_dataset_utils.colorize_obj_mask(obj_label)
        colour_obj_label = cv2.addWeighted(rgb, 0.35, colour_obj_label, 0.65, 0)

        # Img to draw 6-DoF Pose.
        cv2_obj_pose_img = colour_obj_label.copy()

        # AFF
        colour_aff_label = arl_affpose_dataset_utils.colorize_aff_mask(aff_label)
        colour_aff_label = cv2.addWeighted(rgb, 0.35, colour_aff_label, 0.65, 0)

        # OBJ PART
        # obj_label = arl_affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_label)
        # colour_obj_part_label = arl_affpose_dataset_utils.colorize_obj_mask(obj_label)
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
    
    def draw_gt_obj_pose(self, image_idx, verbose=False, project_mesh_on_image=True):

        data = self.get_item(image_idx)

        rgb = data["rgb"]
        depth_16bit = data["depth_16bit"]
        depth_8bit = data["depth_8bit"]
        obj_label = data["obj_label"]
        obj_part_label = data["obj_part_label"]
        cv2_obj_pose_img = data["cv2_obj_pose_img"]
        cv2_obj_part_pose_img = data["cv2_obj_part_pose_img"]
        meta = data["meta"]

        #######################################
        # OBJECT
        #######################################

        obj_ids = np.array(meta['object_class_ids']).flatten()
        label_obj_ids = np.unique(obj_label)[1:]
        label_obj_part_ids = np.unique(obj_part_label)[1:]
        if verbose:
            print('GT obj id: {},\nPred obj id: {},\tPred obj part ids: {}'
                  .format(obj_ids, label_obj_ids, label_obj_part_ids))
        for idx, obj_id in enumerate(obj_ids):
            if obj_id in label_obj_ids:
                obj_id = int(obj_id)
                if verbose:
                    print("\tObject: {}, {}".format(obj_id, self.obj_classes[int(obj_id) - 1]))
                obj_color = arl_affpose_dataset_utils.obj_color_map(obj_id)

                obj_meta_idx = str(1000 + obj_id)[1:]
                obj_r = meta['obj_rotation_' + str(obj_meta_idx)]
                obj_t = meta['obj_translation_' + str(obj_meta_idx)]

                obj_r = np.array(obj_r, dtype=np.float64).reshape(3, 3)
                obj_t = np.array(obj_t, dtype=np.float64).reshape(-1, 3)

                ##################################
                # OCCLUSION
                ##################################

                obj_occlusion_label = np.zeros(shape=(config.OG_HEIGHT, config.OG_WIDTH), dtype=np.uint8)
                obj_part_occlusion_label = np.zeros(shape=(config.OG_HEIGHT, config.OG_WIDTH), dtype=np.uint8)

                ##################################
                # BBOX
                ##################################

                x1, y1, x2, y2 = get_obj_bbox(obj_label.copy(), obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                cv2_obj_pose_img = cv2.rectangle(cv2_obj_pose_img, (x1, y1), (x2, y2), obj_color, 2)
                depth_8bit = cv2.rectangle(depth_8bit, (x1, y1), (x2, y2), 255, 2)

                #######################################
                # ITERATE OVER OBJ PARTS
                #######################################

                obj_part_ids = arl_affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
                if verbose:
                    print('\tobj_part_ids:{}'.format(obj_part_ids))
                for obj_part_id in obj_part_ids:
                    if obj_part_id in label_obj_part_ids:
                        obj_part_id = int(obj_part_id)
                        aff_id = arl_affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                        if verbose:
                            print("\t\tAff: {}, {}".format(aff_id, self.obj_part_classes[int(obj_part_id) - 1]))

                        #######################################
                        # OBJECT POSE
                        #######################################

                        # projecting 3D model to 2D image
                        obj_centered = self.cld_obj_centered[obj_part_id]
                        imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, self.cam_mat, self.cam_dist)
                        if project_mesh_on_image:
                            cv2_obj_pose_img = cv2.polylines(cv2_obj_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                        # modify YCB objects rotation matrix
                        _obj_r = arl_affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_r.copy())

                        # draw pose
                        rotV, _ = cv2.Rodrigues(_obj_r)
                        points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                        axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, self.cam_mat, self.cam_dist)
                        cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                        cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                        cv2_obj_pose_img = cv2.line(cv2_obj_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                        #######################################
                        # OBJ Occlusion
                        #######################################

                        # Draw obj mask.
                        imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, self.og_cam_mat, self.og_cam_dist)
                        obj_projectPoints_img = np.zeros(shape=(config.OG_HEIGHT, config.OG_WIDTH), dtype=np.uint8)
                        obj_projectPoints_img = cv2.polylines(obj_projectPoints_img, np.int32([np.squeeze(imgpts)]), False, (obj_id))
                        masked_obj_projectPoints_img = np.ma.getmaskarray(np.ma.masked_equal(obj_projectPoints_img, obj_id))

                        # get contours.
                        # cv2_masked = np.ma.getmaskarray(np.ma.masked_equal(obj_mask, obj_id)) * cv2_masked
                        res = cv2.findContours(masked_obj_projectPoints_img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                        contours = res[-2]  # for cv2 v3 and v4+ compatibility

                        # get obj mask.
                        obj_occlusion_label = cv2.drawContours(obj_occlusion_label, contours, contourIdx=-1, color=(obj_id), thickness=-1)

                        #######################################
                        # OBJECT PART AFF CENTERED
                        #######################################
                        aff_color = arl_affpose_dataset_utils.aff_color_map(aff_id)

                        obj_part_centered = self.cld_obj_part_centered[obj_part_id]

                        obj_part_id_idx = str(1000 + obj_part_id)[1:]
                        obj_part_r = meta['obj_part_rotation_' + np.str(obj_part_id_idx)]
                        obj_part_t = meta['obj_part_translation_' + np.str(obj_part_id_idx)]

                        #######################################
                        # BBOX
                        #######################################

                        if obj_part_id in arl_affpose_dataset_utils.DRAW_OBJ_PART_POSE:
                            obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_bbox(obj_part_label.copy(), obj_part_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
                            cv2_obj_part_pose_img = cv2.rectangle(cv2_obj_part_pose_img, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), aff_color, 2)
                            depth_8bit = cv2.rectangle(depth_8bit, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), 128, 2)

                        ######################################
                        # 6-DOF POSE
                        #######################################

                        # draw model
                        obj_parts_imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, self.cam_mat, self.cam_dist)
                        if project_mesh_on_image:
                            cv2_obj_part_pose_img = cv2.polylines(cv2_obj_part_pose_img, np.int32([np.squeeze(obj_parts_imgpts)]), False, aff_color)

                        if obj_part_id in arl_affpose_dataset_utils.DRAW_OBJ_PART_POSE:
                            # modify YCB objects rotation matrix
                            _obj_part_r = arl_affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_part_r.copy())
                            # draw pose
                            rotV, _ = cv2.Rodrigues(_obj_part_r)
                            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                            axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, self.cam_mat, self.cam_dist)
                            cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                            cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                            cv2_obj_part_pose_img = cv2.line(cv2_obj_part_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                        #######################################
                        # OBJ Occlusion
                        #######################################

                        # Draw obj mask.
                        imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, self.og_cam_mat, self.og_cam_dist)
                        obj_part_projectPoints_img = np.zeros(shape=(config.OG_HEIGHT, config.OG_WIDTH), dtype=np.uint8)
                        obj_part_projectPoints_img = cv2.polylines(obj_part_projectPoints_img, np.int32([np.squeeze(imgpts)]), False, (obj_id))
                        masked_obj_part_projectPoints_img = np.ma.getmaskarray(np.ma.masked_equal(obj_part_projectPoints_img, obj_id))

                        # get contours.
                        # cv2_masked = np.ma.getmaskarray(np.ma.masked_equal(obj_mask, obj_id)) * cv2_masked
                        res = cv2.findContours(masked_obj_part_projectPoints_img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                        contours = res[-2]  # for cv2 v3 and v4+ compatibility

                        # get obj mask.
                        obj_part_occlusion_label = cv2.drawContours(obj_part_occlusion_label, contours, contourIdx=-1, color=(obj_part_id), thickness=-1)

                #######################################
                # OBJ Occlusion
                #######################################

                # actual.
                masked_obj_label = np.ma.getmaskarray(np.ma.masked_equal(obj_label, obj_id)).astype(np.uint8)
                actual_points = np.count_nonzero(masked_obj_label)

                # projected.
                masked_obj_occlusion_label = np.ma.getmaskarray(np.ma.masked_equal(obj_occlusion_label, obj_id)).astype(np.uint8)
                expected_points = np.count_nonzero(masked_obj_occlusion_label)

                obj_occlusion = 1 - actual_points / (expected_points+1)
                # TODO: standardize occlusion to 1.
                obj_occlusion = max(obj_occlusion, 0.0)
                # print('obj_occlusion: {:.3f}'.format(obj_occlusion))
                meta['obj_occlusion' + str(obj_meta_idx)] = obj_occlusion

                #######################################
                # OBJ Occlusion
                #######################################

                for obj_part_id in obj_part_ids:
                    if obj_part_id in np.unique(obj_part_label):
                        obj_part_id = int(obj_part_id)
                        obj_part_id_idx = str(1000 + obj_part_id)[1:]

                        # actual.
                        masked_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(obj_part_label, obj_part_id)).astype(np.uint8)
                        actual_points = np.count_nonzero(masked_obj_part_label)

                        # projected.
                        masked_obj_part_occlusion_label = np.ma.getmaskarray(np.ma.masked_equal(obj_part_occlusion_label, obj_part_id)).astype(np.uint8)
                        expected_points = np.count_nonzero(masked_obj_part_occlusion_label)

                        obj_part_occlusion = 1 - actual_points / (expected_points+1)
                        # TODO: standardize occlusion to 1.
                        obj_part_occlusion = max(obj_part_occlusion, 0.0)
                        # print('obj_part_occlusion: {:.3f}'.format(obj_part_occlusion))
                        meta['obj_part_occlusion' + str(obj_part_id_idx)] = obj_part_occlusion

                        # cv2.imshow('masked_obj_part_label', masked_obj_part_label * 50)
                        # cv2.imshow('obj_part_occlusion_label', obj_part_occlusion_label * 50)
                        # cv2.waitKey(0)

        #####################
        #####################

        data["meta"] = meta
        data["depth_8bit"] = depth_8bit
        data["cv2_obj_pose_img"] = cv2_obj_pose_img
        data["cv2_obj_part_pose_img"] = cv2_obj_part_pose_img

        return data
