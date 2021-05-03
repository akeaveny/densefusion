import glob
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

#######################################
#######################################

import cfg as config

from utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

def create_pointcloud_from_depth_image(depth,
                                       _cam_scale=config.CAMERA_SCALE,
                                       _cam_fx=config.CAM_FX,
                                       _cam_cx=config.CAM_CX,
                                       _cam_fy=config.CAM_FY,
                                       _cam_cy=config.CAM_CY):

    # INIT
    rows, cols = depth.shape
    xmap = np.array([[j for i in range(cols)] for j in range(rows)])
    ymap = np.array([[i for i in range(cols)] for j in range(rows)])

    depth_masked = depth.flatten()[:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)

    z = depth_masked / _cam_scale
    x = (ymap_masked - _cam_cx) * z / _cam_fx
    y = (xmap_masked - _cam_cy) * z / _cam_fy

    return np.concatenate((y, x, z), axis=1)

def create_masked_pointcloud_from_depth_image(obj_id, label, depth,
                                              _cam_scale=config.CAMERA_SCALE,
                                              _cam_fx=config.CAM_FX,
                                              _cam_cx=config.CAM_CX,
                                              _cam_fy=config.CAM_FY,
                                              _cam_cy=config.CAM_CY):

    # BBOX
    bbox = np.zeros(shape=4)
    x1, y1, x2, y2 = get_obj_bbox(label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
    bbox[0], bbox[1], bbox[2], bbox[3] = x1, y1, x2, y2

    # GET MASK
    mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
    mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, obj_id))
    mask = mask_label * mask_depth

    choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]

    # INIT
    rows, cols = depth.shape
    xmap = np.array([[j for i in range(cols)] for j in range(rows)])
    ymap = np.array([[i for i in range(cols)] for j in range(rows)])

    depth_masked = depth[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    z = depth_masked / _cam_scale
    x = (ymap_masked - _cam_cx) * z / _cam_fx
    y = (xmap_masked - _cam_cy) * z / _cam_fy

    return np.concatenate((x, y, z), axis=1), bbox