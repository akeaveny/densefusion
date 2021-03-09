import glob
import numpy as np

###################################
###################################

from tools.YCB import cfg as config

################################
# TRAIN
################################

image_addr = config.DATA_DIRECTORY_TRAIN + '*/' + '*' + config.RGB_EXT
files = sorted(glob.glob(image_addr))
print(f'\nloaded {len(files)} REAL files for TRAINING')

f_train = open(config.TRAIN_FILE, 'w')
for i, file in enumerate(files):
    str = file.split(config.RGB_EXT)[0]
    str_num = str.split('/')[-1]
    f_train.write(str)
    f_train.write('\n')
print(f'\twrote {i} files')
