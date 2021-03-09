import numpy as np
import torch

#######################################
#######################################

# from pathlib import Path
# DENSEFUSION_ROOT_PATH = Path(__file__).parent.absolute().resolve(strict=True)
# DENSEFUSION_ROOT_PATH = np.str(DENSEFUSION_ROOT_PATH) + '/'

DENSEFUSION_ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

#######################################
#######################################

'''
FRAMEWORK Selection:
'DenseFusion'
'''

# TODO: prelim for naming
# FRAMEWORK           = 'MaskRCNN'
# EXP_DATASET_NAME    = 'UMD_Real_RGB'
# EXP_NUM             = 'v0_Simple_MaskRCNN_128x128'

#######################################
### YCB
#######################################

DATASET_ROOT_PATH = '/data/Akeaveny/Datasets/YCB_Video_Dataset/'

# DATA_DIRECTORY_TRAIN = DATASET_ROOT_PATH + 'data/'
# DATA_DIRECTORY_SYN   = DATASET_ROOT_PATH + 'data_syn/'
# DATA_DIRECTORY_TEST  = DATASET_ROOT_PATH + 'keyframes/'

RGB_EXT    = '-color.png'
DEPTH_EXT  = '-depth.png'
LABEL_EXT  = '-label.png'
META_EXT   = '-meta.mat'
BOX_EXT    = '-box.txt'

#######################################
#######################################

NUM_TEST = 100
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

MATLAB_SCRIPTS_DIR = np.str(DENSEFUSION_ROOT_PATH + '/matlab/')
EVAL_SAVE_FOLDER = DATASET_ROOT_PATH + 'pred/'

TEST_SAVE_FOLDER = DATASET_ROOT_PATH + 'test_densefusion/'

#######################################
### DATASET CONFIG
#######################################

CLASSES_FILE   = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/classes.txt'
CLASS_IDS_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/class_ids.txt'

TRAIN_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/train_data_list.txt'
TEST_FILE  = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/test_data_list.txt'

#######################################
#######################################

NUM_OBJECT_CLASSES = 17 + 1         # 1 is for the background

IMG_MEAN   = [98.92739272, 66.78827961, 71.00867078, 135.8963934]
IMG_STD    = [26.53540375, 31.51117582, 31.75977128, 38.23637208]

#######################################
# CAMERA CONFIGS
#######################################

CAM_CX_1 = 312.9869
CAM_CY_1 = 241.3109
CAM_FX_1 = 1066.778
CAM_FY_1 = 1067.487

CAM_CX_2 = 323.7872
CAM_CY_2 = 279.6921
CAM_FX_2 = 1077.836
CAM_FY_2 = 1078.189

HEIGHT, WIDTH = 640, 480
ORIGINAL_SIZE = (HEIGHT, WIDTH)
RESIZE        = (int(HEIGHT/1), int(WIDTH/1))

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

XMAP = np.array([[j for i in range(HEIGHT)] for j in range(WIDTH)])
YMAP = np.array([[i for i in range(HEIGHT)] for j in range(WIDTH)])

#######################################
#######################################

NUM_PT = 1000
NUM_PT_MESH_SMALL = 500
NUM_PT_MESH_LARGE = 2600

