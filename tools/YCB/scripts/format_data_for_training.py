import glob
import numpy as np

#######################################
#######################################

import sys
sys.path.append('../../../')

#######################################
#######################################

import tools.YCB.cfg as config

################################
# TRAIN
################################
print('\n-------- TRAIN --------')

files = open(config.OG_TRAIN_FILE)
files = np.loadtxt(files, dtype=np.str)
print('Loaded {} Images'.format(len(files)))

# select random test images
# np.random.seed(0)
# total_idx = np.arange(0, len(files), 1)
# train_idx = np.random.choice(total_idx, size=int(config.NUM_TRAIN), replace=False)
# train_files = np.array(files)[train_idx]
# print("Chosen Train: {}".format(len(train_files)))

idx = np.arange(0, int(16187), 1)
train_files = files[idx]

f_train = open(config.TRAIN_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(train_files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TRAIN + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_train.write(str_num)
    f_train.write('\n')
print('wrote {} files'.format(i+1))