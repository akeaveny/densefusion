import glob
import numpy as np
import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

import tools.Elevator.cfg as config

###################################
# ARL
###################################

class_file    = open(config.CLASSES_FILE)
class_id_file = open(config.CLASS_IDS_FILE)
class_IDs     = np.loadtxt(class_id_file, dtype=np.int32)
print("class_IDs: ", class_IDs)

################################
# TRAIN
################################
print('\n-------- TRAIN --------')

# /data/Akeaveny/Datasets/Elevator/Real/train/rgb/000000.png
gt_label_addr = config.DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
files = sorted(glob.glob(gt_label_addr))
print('Loaded {} Images'.format(len(files)))

f_train = open(config.TRAIN_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TRAIN + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_train.write(str_num)
    f_train.write('\n')
print('wrote {} files'.format(i+1))

################################
# VAL
################################
print('\n-------- VAL --------')

gt_label_addr = config.DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
files = sorted(glob.glob(gt_label_addr))
print('Loaded {} Images'.format(len(files)))

f_val = open(config.VAL_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_VAL + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_val.write(str_num)
    f_val.write('\n')
print('wrote {} files'.format(i+1))

################################
# TEST
################################
print('\n-------- TEST --------')

gt_label_addr = config.DATA_DIRECTORY_TEST + 'rgb/' + '*' + config.RGB_EXT
files = sorted(glob.glob(gt_label_addr))
print('Loaded {} Images'.format(len(files)))

f_test = open(config.TEST_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TEST + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_test.write(str_num)
    f_test.write('\n')
print('wrote {} files'.format(i+1))