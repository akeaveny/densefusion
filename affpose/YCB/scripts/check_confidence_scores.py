import os
import glob

import copy
import random

import numpy as np
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt

import scipy.io as scio

#######################################
#######################################

import sys
sys.path.append('../../../')

from affpose.YCB import cfg as config
from affpose.YCB.utils.dataset import ycb_dataset_utils

#######################################
#######################################

def main():

    # MATLAB RESULTS DIR.
    file_path = config.EVAL_FOLDER_DF_ITERATIVE + '/*.mat'
    files = sorted(glob.glob(file_path))

    pred_class_ids = np.zeros(shape=(len(files), 10))
    pred_c = np.zeros(shape=(len(files), 10))
    # for idx in range(len(files)):
    for file_idx, file in enumerate(files):
        meta = scio.loadmat(file)
        _class_ids = meta['class_ids'].reshape(-1)
        _confidence = meta['confidence'].reshape(-1)
        for class_idx in range(len(_class_ids)):
            pred_class_ids[file_idx, class_idx] = _class_ids[class_idx]
            pred_c[file_idx, class_idx] = _confidence[class_idx]
    # flatten arrays and find non zero idxs.
    pred_class_ids = pred_class_ids.reshape(-1)
    pred_c = pred_c.reshape(-1)
    non_zero_idx = np.nonzero(pred_class_ids)
    pred_class_ids = pred_class_ids[non_zero_idx]
    pred_c = pred_c[non_zero_idx]

    # now we want to plot predictions
    for obj_id in range(1, config.NUM_OBJECTS+1):
        # get pred c for class.
        row_idxs = np.argwhere(pred_class_ids == obj_id).reshape(-1)
        _pred_c = np.sort(pred_c[row_idxs])

        fig = plt.figure(obj_id)
        plt.title(f'Object Id: {obj_id}, {ycb_dataset_utils.map_obj_id_to_name(obj_id)}', fontsize=10)
        plt.xlabel('$Confidence$', fontsize=10)
        plt.ylabel('$Frequency$', fontsize=10)
        plt.xlim(0, 1.15)
        # plotting configs
        _color =  ycb_dataset_utils.obj_color_map(idx=obj_id)
        color = [_color[0]/255, _color[1]/255, _color[2]/255, 0.75]

        # get histogram.
        mean = np.mean(_pred_c)
        std_dev = np.std(_pred_c)
        print(f'Object Id: {obj_id},\t\t Num: {len(row_idxs)}'
              f'\t\t mean:{mean:.5f},\t\t std_dev:{std_dev:.5f},'
              f'\t\t Name: {ycb_dataset_utils.map_obj_id_to_name(obj_id)}')

        # plot data.
        # plt.plot(range(len(row_idxs)), _pred_c, color=color, label=f'{arl_affpose_dataset_utils.map_obj_id_to_name(obj_id)}')
        plt.hist(_pred_c, bins=10, color=color, label=f'{ycb_dataset_utils.map_obj_id_to_name(obj_id)}')
    # plt.show()

if __name__ == '__main__':
    main()