import numpy as np

import torch
import torchvision.transforms as transforms

#######################################
# DenseFusion
#######################################

ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

CLASSES_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/classes.txt'
CLASS_IDS_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/classes_ids.txt'
OBJ_PART_CLASSES_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes.txt'
OBJ_PART_CLASS_IDS_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_ids.txt'
OBJ_PART_CLASSES_FILE_TRAIN   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_train.txt'
OBJ_PART_CLASS_IDS_FILE_TRAIN = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_ids_train.txt'

TRAIN_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/train_list.txt'
VAL_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/val_list.txt'
TEST_FILE  = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/test_list.txt'

# Trained models
# PRE_TRAINED_MODEL        = ROOT_PATH + 'trained_models/arl_vicon_retrain/pose_model_6_0.011263955466539667.pth'
# PRE_TRAINED_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_vicon_retrain/pose_refine_model_93_0.006630927582543332.pth'

# MATLAB
EVAL_FOLDER_GT           = ROOT_PATH + 'tools/ARLAffPose/matlab/results/gt'
EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'tools/ARLAffPose/matlab/results/df_wo_refine'
EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'tools/ARLAffPose/matlab/results/df_iterative'

#######################################
# Dataset Prelims
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'
TEST_DENSEFUSION_FOLDER = ROOT_DATA_PATH + 'test_densefuion/'

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

WIDTH, HEIGHT = 1280, 720
ORIGINAL_SIZE = (WIDTH, HEIGHT)
RESIZE        = (int(WIDTH/1), int(HEIGHT/1))
CROP_SIZE     = (int(640), int(480)) # (int(640), int(480)) or (int(1280), int(720))
WIDTH, HEIGHT = CROP_SIZE[0], CROP_SIZE[1]
MIN_SIZE = MAX_SIZE = 640

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

X_SCALE = CROP_SIZE[0] / ORIGINAL_SIZE[0]
Y_SCALE = CROP_SIZE[1] / ORIGINAL_SIZE[1]

CAMERA_SCALE = 1000

# Real
# CAM_CX = 652.26074
# CAM_CY = 335.50336
CAM_CX = 652.26074 * X_SCALE
CAM_CY = 335.50336 * Y_SCALE
CAM_FX = 680.72644
CAM_FY = 680.72644

# Syn
# CAM_CX = 653.5618286132812
# CAM_CY = 338.541748046875
# # CAM_CX = 653.5618286132812 * X_SCALE
# # CAM_CY = 338.541748046875  * Y_SCALE
# CAM_FX = 682.7849731445312
# CAM_FY = 682.7849731445312

XMAP = np.array([[j for i in range(WIDTH)] for j in range(HEIGHT)])
YMAP = np.array([[i for i in range(WIDTH)] for j in range(HEIGHT)])

CAM_MAT = np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]])
CAM_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#######################################
#######################################

NUM_OBJECTS = 11
SYM_OBJECTS = [1, 4, 7, 11]

IMG_MEAN   = [0.485, 0.456, 0.406]
IMG_STD    = [0.229, 0.224, 0.225]

NORM = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

NUM_PT = 500

NUM_PT_MIN = 50
NUM_PT_MESH_SMALL = 500
NUM_PT_MESH_LARGE = 800

REFINE_ITERATIONS = 2
BATCH_SIZE = 1



