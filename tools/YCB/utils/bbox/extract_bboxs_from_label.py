
import yaml
import numpy as np

#######################################
#######################################

from tools.utils import helper_utils

from tools.YCB import cfg as config
from tools.YCB.utils.dataset import ycb_dataset_utils

###########################################################
# bbox
###########################################################

def get_posecnn_bbox(posecnn_rois, pred_to_gt_idx):
    rmin = int(posecnn_rois[pred_to_gt_idx][3]) + 1
    rmax = int(posecnn_rois[pred_to_gt_idx][5]) - 1
    cmin = int(posecnn_rois[pred_to_gt_idx][2]) + 1
    cmax = int(posecnn_rois[pred_to_gt_idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(config.BORDER_LIST)):
        if r_b > config.BORDER_LIST[tt] and r_b < config.BORDER_LIST[tt + 1]:
            r_b = config.BORDER_LIST[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(config.BORDER_LIST)):
        if c_b > config.BORDER_LIST[tt] and c_b < config.BORDER_LIST[tt + 1]:
            c_b = config.BORDER_LIST[tt + 1]
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
    if rmax > config.WIDTH:
        delt = rmax - config.WIDTH
        rmax = config.WIDTH
        rmin -= delt
    if cmax > config.HEIGHT:
        delt = cmax - config.HEIGHT
        cmax = config.HEIGHT
        cmin -= delt
    return rmin, rmax, cmin, cmax

###########################################################
# bbox
###########################################################

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(config.BORDER_LIST)):
        if r_b > config.BORDER_LIST[tt] and r_b < config.BORDER_LIST[tt + 1]:
            r_b = config.BORDER_LIST[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(config.BORDER_LIST)):
        if c_b > config.BORDER_LIST[tt] and c_b < config.BORDER_LIST[tt + 1]:
            c_b = config.BORDER_LIST[tt + 1]
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
    if rmax > config.WIDTH:
        delt = rmax - config.WIDTH
        rmax = config.WIDTH
        rmin -= delt
    if cmax > config.HEIGHT:
        delt = cmax - config.HEIGHT
        cmax = config.HEIGHT
        cmin -= delt
    return rmin, rmax, cmin, cmax

###########################################################
# obj bbox
###########################################################

def get_obj_bbox(mask, obj_id, img_width, img_length, border_list):

    ####################
    ## affordance id
    ####################

    rows = np.any(mask==obj_id, axis=1)
    cols = np.any(mask==obj_id, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    y2 += 1
    x2 += 1
    r_b = y2 - y1
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = x2 - x1
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((y1 + y2) / 2), int((x1 + x2) / 2)]
    y1 = center[0] - int(r_b / 2)
    y2 = center[0] + int(r_b / 2)
    x1 = center[1] - int(c_b / 2)
    x2 = center[1] + int(c_b / 2)
    if y1 < 0:
        delt = -y1
        y1 = 0
        y2 += delt
    if x1 < 0:
        delt = -x1
        x1 = 0
        x2 += delt
    if y2 > img_width:
        delt = y2 - img_width
        y2 = img_width
        y1 -= delt
    if x2 > img_length:
        delt = x2 - img_length
        x2 = img_length
        x1 -= delt
    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    # cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return x1, y1, x2, y2