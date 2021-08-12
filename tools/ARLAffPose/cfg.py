import numpy as np

#######################################
# DenseFusion
#######################################

ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

CLASSES_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/classes.txt'
CLASS_IDS_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/classes_ids.txt'
OBJ_PART_CLASSES_FILE   = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes.txt'
OBJ_PART_CLASS_IDS_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_ids.txt'
OBJ_PART_CLASSES_FILE_TRAIN   = OBJ_PART_CLASSES_FILE # ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_train.txt'
OBJ_PART_CLASS_IDS_FILE_TRAIN = OBJ_PART_CLASS_IDS_FILE # ROOT_PATH + 'datasets/arl_affpose/dataset_config/obj_part_classes_ids_train.txt'

TRAIN_FILE = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/train_list.txt'
VAL_FILE  = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/val_list.txt'
TEST_FILE  = ROOT_PATH + 'datasets/arl_affpose/dataset_config/data_lists/test_list.txt'

# Trained models
# PRE_TRAINED_MODEL        = ROOT_PATH + 'trained_models/arl_affpose_obj/real_and_syn_v2/pose_model_18_0.012826319938234909.pth'
# PRE_TRAINED_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_affpose_obj/real_and_syn_v2/pose_refine_model_43_0.01054153353055539.pth'
PRE_TRAINED_MODEL        = ROOT_PATH + 'trained_models/arl_affpose_obj/real_and_syn_v3/pose_model_18_0.012060843364452012.pth'
PRE_TRAINED_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_affpose_obj/real_and_syn_v3/pose_refine_model_37_0.010551110985974044.pth'
# PRE_TRAINED_AFF_MODEL        = ROOT_PATH + 'trained_models/arl_affpose_aff/real_and_syn_v2/pose_model_14_0.012333551328883181.pth'
# PRE_TRAINED_AFF_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_affpose_aff/real_and_syn_v2/pose_refine_model_27_0.010844891724854883.pth'
PRE_TRAINED_AFF_MODEL        = ROOT_PATH + 'trained_models/arl_affpose_aff/real_and_syn_v3/pose_model_16_0.012952367823778796.pth'
PRE_TRAINED_AFF_REFINE_MODEL = ROOT_PATH + 'trained_models/arl_affpose_aff/real_and_syn_v3/pose_refine_model_55_0.010159908071935377.pth'

# MATLAB
OBJ_EVAL_FOLDER_GT           = ROOT_PATH + 'tools/ARLAffPose/matlab/results/obj/gt'
OBJ_EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'tools/ARLAffPose/matlab/results/obj/df_wo_refine'
OBJ_EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'tools/ARLAffPose/matlab/results/obj/df_iterative'
AFF_EVAL_FOLDER_GT           = ROOT_PATH + 'tools/ARLAffPose/matlab/results/aff/gt'
AFF_EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'tools/ARLAffPose/matlab/results/aff/df_wo_refine'
AFF_EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'tools/ARLAffPose/matlab/results/aff/df_iterative'

#######################################
# Dataset Prelims
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'
TEST_DENSEFUSION_FOLDER = ROOT_DATA_PATH + 'test_densefusion/'

SELECT_EVERY_ITH_FRAME = 3  # similar to YCB-Video Dataset

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

WIDTH, HEIGHT = 1280, 720
ORIGINAL_SIZE = (WIDTH, HEIGHT)
RESIZE        = (int(WIDTH/1), int(HEIGHT/1))
CROP_SIZE     = (int(640), int(640)) # (int(640), int(640)) or (int(1280), int(720))
WIDTH, HEIGHT = CROP_SIZE[0], CROP_SIZE[1]
MIN_SIZE = np.min([WIDTH, HEIGHT])
MAX_SIZE = np.max([WIDTH, HEIGHT])

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

XMAP = np.array([[j for i in range(WIDTH)] for j in range(HEIGHT)])
YMAP = np.array([[i for i in range(WIDTH)] for j in range(HEIGHT)])

X_SCALE = CROP_SIZE[0] / ORIGINAL_SIZE[0]
Y_SCALE = CROP_SIZE[1] / ORIGINAL_SIZE[1]

# Real
# CAM_CX = 652.26074 * X_SCALE
# CAM_CY = 335.50336 * Y_SCALE
# CAM_FX = 680.72644
# CAM_FY = 680.72644

# Syn
# CAM_CX = 653.5618286132812 * X_SCALE
# CAM_CY = 338.541748046875  * Y_SCALE
# CAM_FX = 682.7849731445312
# CAM_FY = 682.7849731445312

# # ARL
# CAM_CX = 615.583 * X_SCALE
# CAM_CY = 359.161 * Y_SCALE
# CAM_FX = 739.436
# CAM_FY = 739.436

# CAM_MAT = np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]])
# CAM_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#######################################
#######################################

NUM_IMAGES = 90000
NUM_TRAIN  = NUM_IMAGES
NUM_TEST    = 1500

NUM_OBJECTS = 11
NUM_OBJECTS_PARTS = 25
SYM_OBJECTS = [1, 4, 7, 11]

# REAL
IMG_MEAN   = [103.15870247/255, 102.75600649/255, 83.28931782/255]
IMG_STD    = [57.65692223/255, 64.30550654/255, 45.79033007/255]

# REAL AND SYN
# IMG_MEAN   = [132.08893337/255, 103.38624212/255, 84.43098558/255]
# IMG_STD    = [46.52742011/255, 54.07110987/255, 39.97097183/255]

NUM_PT = 500

NUM_PT_MIN = 50
NUM_PT_MESH_SMALL = 250 # 500
NUM_PT_MESH_LARGE = 500 # 2600

FRONT_NUM = 2

REFINE_ITERATIONS = 4
BATCH_SIZE = 1

PRED_C_THRESHOLD = 0.0



