import numpy as np

import torch
import torchvision.transforms as transforms

#######################################
#######################################

DENSEFUSION_ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

# PRE_TRAINED_MODEL = DENSEFUSION_ROOT_PATH + '/trained_models/pretrained_ycb/pose_model_26_0.012863246640872631.pth'
# PRE_TRAINED_REFINE_MODEL = DENSEFUSION_ROOT_PATH + '/trained_models/pretrained_ycb/pose_refine_model_69_0.009449292959118935.pth'

PRE_TRAINED_MODEL = DENSEFUSION_ROOT_PATH + 'trained_models/pretrained_ycb/pose_model_26_0.012863246640872631.pth'
PRE_TRAINED_REFINE_MODEL = DENSEFUSION_ROOT_PATH + 'trained_models/pretrained_ycb/pose_refine_model_69_0.009449292959118935.pth'

#######################################
### YCB
#######################################

DATASET_ROOT_PATH = '/data/Akeaveny/Datasets/YCB_Video_Dataset/'

RGB_EXT     = '-color.png'
DEPTH_EXT   = '-depth.png'
LABEL_EXT   = '-label.png'
META_EXT    = '-meta.mat'
BOX_EXT     = '-box.txt'

POSECNN_EXT = '.mat'

#######################################
#######################################

NUM_TEST = 100
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

TEST_SAVE_FOLDER = DATASET_ROOT_PATH + 'test_densefusion/'

MATLAB_SCRIPTS_DIR = np.str(DENSEFUSION_ROOT_PATH + '/matlab/')

EVAL_FOLDER_GT           = DENSEFUSION_ROOT_PATH + 'tools/YCB/matlab/results/gt'
EVAL_FOLDER_POSECNN      = DENSEFUSION_ROOT_PATH + 'tools/YCB/matlab/results/posecnn'
EVAL_FOLDER_DF_WO_REFINE = DENSEFUSION_ROOT_PATH + 'tools/YCB/matlab/results/df_wo_refine'
EVAL_FOLDER_DF_ITERATIVE = DENSEFUSION_ROOT_PATH + 'tools/YCB/matlab/results/df_iterative'

#######################################
### DATASET CONFIG
#######################################

CLASSES_FILE   = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/classes.txt'
CLASS_IDS_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/class_ids.txt'

# OG_TRAIN_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/train_data_list.txt'
# OG_TEST_FILE  = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/test_data_list.txt'
TRAIN_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/train_data_list.txt'
TEST_FILE  = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/keyframe.txt'

YCB_TOOLBOX_CONFIG = DENSEFUSION_ROOT_PATH + 'YCB_Video_toolbox/results_PoseCNN_RSS2018/'

#######################################
#######################################

NUM_OBJECTS = 21

IMG_MEAN   = [0.485, 0.456, 0.406]
IMG_STD    = [0.229, 0.224, 0.225]

NORM = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

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

CAM_SCALE = 10000.0

HEIGHT, WIDTH = 640, 480
ORIGINAL_SIZE = (HEIGHT, WIDTH)
RESIZE        = (int(HEIGHT/1), int(WIDTH/1))

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

XMAP = np.array([[j for i in range(HEIGHT)] for j in range(WIDTH)])
YMAP = np.array([[i for i in range(HEIGHT)] for j in range(WIDTH)])

#######################################
#######################################

NUM_IMAGES = 50000
NUM_TRAIN  = NUM_IMAGES
NUM_VAL    = 2949

NUM_PT = 500
NUM_PT_MESH_SMALL = 500
NUM_PT_MESH_LARGE = 2600

REFINE_ITERATIONS = 2
BATCH_SIZE = 1

