import glob
import numpy as np

#######################################
#######################################

import tools.ARLVicon.cfg as config

################################
# TRAIN
################################
print('\n-------- TRAIN --------')

# real
real_gt_label_addr = config.DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
real_train_files = sorted(glob.glob(real_gt_label_addr))
real_gt_label_addr = config.DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
real_val_files = sorted(glob.glob(real_gt_label_addr))
# combined
real_files = np.array(np.hstack([real_train_files, real_val_files]))
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

# real
real_gt_label_addr = config.DATA_DIRECTORY_TEST + 'rgb/' + '*' + config.RGB_EXT
files = sorted(glob.glob(real_gt_label_addr))
print('Loaded {} Images'.format(len(files)))

f_test = open(config.TEST_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TEST + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_test.write(str_num)
    f_test.write('\n')
print('wrote {} files'.format(i+1))