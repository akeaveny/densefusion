import glob
import numpy as np

#######################################
#######################################

import tools.ARLVicon.cfg as config

###################################
# PRELIM
###################################

class_file    = open(config.CLASSES_FILE)
class_id_file = open(config.CLASS_IDS_FILE)
class_IDs     = np.loadtxt(class_id_file, dtype=np.int32)
print("class_IDs: ", class_IDs)

################################
# TRAIN
################################
print('\n-------- TRAIN --------')

# real
real_gt_label_addr = config.DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
real_files = sorted(glob.glob(real_gt_label_addr))
# syn
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_TRAIN + 'rgb/' + '*' + config.RGB_EXT
syn_files = sorted(glob.glob(syn_gt_label_addr))
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
# VAL
################################
print('\n-------- VAL --------')

# real
real_gt_label_addr = config.DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
real_files = sorted(glob.glob(real_gt_label_addr))
# syn
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_VAL + 'rgb/' + '*' + config.RGB_EXT
syn_files = sorted(glob.glob(syn_gt_label_addr))
# combined
files = np.array(np.hstack([real_files, syn_files]))
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

# real
real_gt_label_addr = config.DATA_DIRECTORY_TEST + 'rgb/' + '*' + config.RGB_EXT
real_files = sorted(glob.glob(real_gt_label_addr))
# syn
syn_gt_label_addr = config.SYN_DATA_DIRECTORY_TEST + 'rgb/' + '*' + config.RGB_EXT
syn_files = sorted(glob.glob(syn_gt_label_addr))
# combined
files = real_files # np.array(np.hstack([real_files, syn_files]))
print('Loaded {} Images'.format(len(files)))

f_test = open(config.TEST_FILE, 'w')
# ===================== train ====================
for i, file in enumerate(files):
    # str_num = file.split(config.RGB_EXT)[0].split(config.DATA_DIRECTORY_TEST + 'rgb/')[1]
    str_num = file.split(config.RGB_EXT)[0]
    f_test.write(str_num)
    f_test.write('\n')
print('wrote {} files'.format(i+1))