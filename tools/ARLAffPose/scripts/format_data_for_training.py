import glob
import numpy as np

#######################################
#######################################

import sys
sys.path.append('../../../')

#######################################
#######################################

import tools.ARLAffPose.cfg as config

################################
# TRAIN
################################
print('\n-------- TRAIN --------')

# real
real_gt_label_addr = config.DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
real_files = np.sort(np.array(glob.glob(real_gt_label_addr)))
# TODO: selecting every ith images.
total_idx = np.arange(0, len(real_files), config.SELECT_EVERY_ITH_FRAME)
real_files = np.sort(np.array(real_files)[total_idx])
print('Loaded {} Images'.format(len(real_files)))

# syn
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
syn_train_files = np.sort(np.array(glob.glob(syn_gt_label_addr)))
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
syn_val_files = np.sort(np.array(glob.glob(syn_gt_label_addr)))
syn_files = np.sort(np.array(np.hstack([syn_train_files, syn_val_files])))
# TODO: selecting every ith images.
total_idx = np.arange(0, len(syn_files), config.SELECT_EVERY_ITH_FRAME*2)
syn_files = np.sort(np.array(syn_files)[total_idx])
print('Loaded {} Images'.format(len(syn_files)))

# combined
files = np.array(np.hstack([real_files, syn_files]))
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
# VAL
################################
print('\n-------- VAL --------')

# real
real_gt_label_addr = config.DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
real_files = np.sort(np.array(glob.glob(real_gt_label_addr)))
# TODO: selecting every ith images.
total_idx = np.arange(0, len(real_files), config.SELECT_EVERY_ITH_FRAME*2)
files = np.sort(np.array(real_files)[total_idx])
print("Chosen Val: {}".format(len(files)))

f_val = open(config.VAL_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TRAIN + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_val.write(str_num)
    f_val.write('\n')
print('wrote {} files'.format(i+1))

################################
# TEST
################################
print('\n-------- TEST --------')

# test
real_gt_label_addr = config.DATA_DIRECTORY_TEST + 'rgb/' + '*' + config.RGB_EXT
real_files = np.sort(np.array(glob.glob(real_gt_label_addr)))
# selecting every ith images.
total_idx = np.arange(0, len(real_files), config.SELECT_EVERY_ITH_FRAME*2)
files = np.array(real_files)[total_idx]
print("Chosen Test: {}".format(len(files)))

f_test = open(config.TEST_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TRAIN + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_test.write(str_num)
    f_test.write('\n')
print('wrote {} files'.format(i+1))