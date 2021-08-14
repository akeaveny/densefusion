import numpy as np

#######################################
# DenseFusion
#######################################

ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

CLASSES_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/classes.txt'
CLASS_IDS_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/classes_ids.txt'
OBJ_PART_CLASSES_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes.txt'
OBJ_PART_CLASS_IDS_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_ids.txt'
OBJ_PART_CLASSES_FILE_TRAIN   = OBJ_PART_CLASSES_FILE  # ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_train.txt'
OBJ_PART_CLASS_IDS_FILE_TRAIN = OBJ_PART_CLASS_IDS_FILE  # ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_ids_train.txt'

TRAIN_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/train_list.txt'
VAL_FILE  = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/val_list.txt'
TEST_FILE  = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/test_list.txt'

# Trained models
PRE_TRAINED_MODEL        = ROOT_PATH + 'trained_models/arl_affpose_obj/real_and_syn_v3/pose_model_18_0.012060843364452012.pth'
PRE_TRAINED_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_affpose_obj/real_and_syn_v3/pose_refine_model_37_0.010551110985974044.pth'
PRE_TRAINED_AFF_MODEL        = ROOT_PATH + 'trained_models/arl_affpose_aff/real_and_syn_v3/pose_model_16_0.012952367823778796.pth'
PRE_TRAINED_AFF_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_affpose_aff/real_and_syn_v3/pose_refine_model_55_0.010159908071935377.pth'

# MATLAB
OBJ_EVAL_FOLDER_GT           = ROOT_PATH + 'affpose/ARLAffPose/matlab/obj/results/gt'
OBJ_EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'affpose/ARLAffPose/matlab/obj/results/df_wo_refine'
OBJ_EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'affpose/ARLAffPose/matlab/obj/results/df_iterative'
AFF_EVAL_FOLDER_GT           = ROOT_PATH + 'affpose/ARLAffPose/matlab/aff/results/gt'
AFF_EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'affpose/ARLAffPose/matlab/aff/results/df_wo_refine'
AFF_EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'affpose/ARLAffPose/matlab/aff/results/df_iterative'

#######################################
# Dataset Prelims
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'
TEST_DENSEFUSION_FOLDER = ROOT_DATA_PATH + 'test_densefusion/'

SELECT_EVERY_ITH_FRAME_TRAIN = 3  # similar to YCB-Video Dataset
SELECT_EVERY_ITH_FRAME_TEST = 6

# REAL
DATA_DIRECTORY       = ROOT_DATA_PATH + 'Real/'
DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
DATA_DIRECTORY_VAL   = DATA_DIRECTORY + 'val/'
DATA_DIRECTORY_TEST  = DATA_DIRECTORY + 'test/'

# SYN
SYN_DATA_DIRECTORY       = ROOT_DATA_PATH + 'Syn/'
SYN_DATA_DIRECTORY_TRAIN = SYN_DATA_DIRECTORY + 'train/'
SYN_DATA_DIRECTORY_VAL   = SYN_DATA_DIRECTORY + 'val/'
SYN_DATA_DIRECTORY_TEST  = SYN_DATA_DIRECTORY + 'test/'

RGB_EXT            = '.png'
DEPTH_EXT          = '_depth.png'
OBJ_LABEL_EXT      = '_obj_label.png'
OBJ_PART_LABEL_EXT = '_obj_part_labels.png'
AFF_LABEL_EXT      = '_aff_label.png'
META_EXT           = '_meta.mat'

#######################################
# ZED CAMERA
#######################################

CAMERA_SCALE = 1000

OG_WIDTH, OG_HEIGHT = 1280, 720
WIDTH, HEIGHT = 1280, 720
ORIGINAL_SIZE = (WIDTH, HEIGHT)
RESIZE = (int(WIDTH/1), int(HEIGHT/1))
CROP_SIZE = (int(640), int(640))
WIDTH, HEIGHT = CROP_SIZE[0], CROP_SIZE[1]
MIN_SIZE = np.min([WIDTH, HEIGHT])
MAX_SIZE = np.max([WIDTH, HEIGHT])

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

XMAP = np.array([[j for i in range(WIDTH)] for j in range(HEIGHT)])
YMAP = np.array([[i for i in range(WIDTH)] for j in range(HEIGHT)])

X_SCALE = CROP_SIZE[0] / ORIGINAL_SIZE[0]
Y_SCALE = CROP_SIZE[1] / ORIGINAL_SIZE[1]

#######################################
#######################################

NUM_OBJECTS = 11
NUM_OBJECTS_PARTS = 25
SYM_OBJECTS = [1, 4, 7, 11]
SYM_AFF_OBJECTS = [1, 2, 7, 8, 14, 15, 24, 25]

# Real, test images.
IMG_MEAN   = [164.31746134/255, 157.29184788/255, 139.49621539/255]
IMG_STD    = [49.55041067/255, 64.919871/255, 64.19674351/255]

NUM_PT = 500
NUM_PT_MIN = 50
NUM_PT_MESH_SMALL = 250  # 500
NUM_PT_MESH_LARGE = 500  # 2600

FRONT_NUM = 2

REFINE_ITERATIONS = 2
BATCH_SIZE = 1

PRED_C_THRESHOLD = 0.0



