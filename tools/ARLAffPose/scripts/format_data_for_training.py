import glob
import numpy as np

#######################################
#######################################

import sys
sys.path.append('../../../')

#######################################
#######################################

import tools.ARLAffPose.cfg as config

#######################################
#######################################

# select random test images
# np.random.seed(0)
# total_idx = np.arange(0, len(files), 1)
# train_idx = np.random.choice(total_idx, size=int(config.NUM_TRAIN), replace=False)
# # train_idx = np.random.choice(total_idx, size=int(len(train_files)), replace=False)
# files = np.array(files)[train_idx]
# print("Chosen Train: {}".format(len(files)))

################################
# TRAIN
################################
print('\n-------- TRAIN --------')

# real
real_gt_label_addr = config.DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
real_train_files = sorted(glob.glob(real_gt_label_addr))
# real_gt_label_addr = config.DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
# real_val_files = sorted(glob.glob(real_gt_label_addr))
# combined
real_files = real_train_files # np.array(np.hstack([real_train_files, real_val_files]))
print('Loaded {} Images'.format(len(real_files)))

# syn
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
syn_train_files = sorted(glob.glob(syn_gt_label_addr))
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
syn_val_files = sorted(glob.glob(syn_gt_label_addr))
# combined
syn_files = np.array(np.hstack([syn_train_files, syn_val_files]))
print('Loaded {} Images'.format(len(syn_files)))

# combined
files = np.array(np.hstack([real_files, syn_files]))
print('Loaded {} Images'.format(len(files)))

# select every ith images
total_idx = np.arange(0, len(files), config.SELECT_EVERY_ITH_FRAME)
files = np.array(files)[total_idx]
print("Chosen Train: {}".format(len(files)))

f_train = open(config.TRAIN_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TRAIN + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_train.write(str_num)
    f_train.write('\n')
print('wrote {} files'.format(i+1))

################################
# TEST
################################
print('\n-------- TEST --------')

# test
real_gt_label_addr = config.DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
files = sorted(glob.glob(real_gt_label_addr))
print('Loaded {} Images from {}'.format(len(files), real_gt_label_addr))

# select every ith images
total_idx = np.arange(0, len(files), config.SELECT_EVERY_ITH_FRAME)
files = np.array(files)[total_idx]
print("Chosen Train: {}".format(len(files)))

f_test = open(config.VAL_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(con
    # print('wrote {} files'.format(i+1))fig.RGB_EXT)[0].split(config.DATA_DIRECTORY_TEST + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_test.write(str_num)
    f_test.write('\n')
print('wrote {} files'.format(i + 1))