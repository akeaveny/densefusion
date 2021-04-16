import numpy as np

#######################################
# DenseFusion
#######################################

ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

# /home/akeaveny/git/DenseFusion/datasets/elevator/dataset_config/classes.txt
CLASSES_FILE   = ROOT_PATH + 'datasets/elevator/dataset_config/classes.txt'
CLASS_IDS_FILE = ROOT_PATH + 'datasets/elevator/dataset_config/classes_ids.txt'

# /home/akeaveny/git/DenseFusion/datasets/elevator/dataset_config/data_lists/train_list.txt
TRAIN_FILE = ROOT_PATH + 'datasets/elevator/dataset_config/data_lists/train_list.txt'
VAL_FILE   = ROOT_PATH + 'datasets/elevator/dataset_config/data_lists/val_list.txt'
TEST_FILE  = ROOT_PATH + 'datasets/elevator/dataset_config/data_lists/test_list.txt'

#######################################
###
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/Elevator/'
TEST_DENSEFUSION_FOLDER = ROOT_DATA_PATH + 'test_densefuion/'

### REAL
DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'

RGB_EXT       = ".png"
DEPTH_EXT     = "_depth.png"
LABEL_EXT     = "_label.png"
META_EXT      = "_meta.mat"

#######################################
# ZED CAMERA
#######################################

WIDTH, HEIGHT = 672, 376
ORIGINAL_SIZE = (WIDTH, HEIGHT)
RESIZE        = (int(WIDTH/1), int(HEIGHT/1))
INPUT_SIZE    = (int(384), int(384))
WIDTH, HEIGHT = INPUT_SIZE[0], INPUT_SIZE[1]
MIN_SIZE = MAX_SIZE = 384

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

X_SCALE = INPUT_SIZE[0] / ORIGINAL_SIZE[0]
Y_SCALE = INPUT_SIZE[1] / ORIGINAL_SIZE[1]

CAMERA_SCALE = 1000
# CAM_CX = 341.276
# CAM_CY = 175.296
CAM_CX = 341.276 * X_SCALE
CAM_CY = 175.296 * Y_SCALE
CAM_FX = 338.546630859375
CAM_FY = 338.546630859375

XMAP = np.array([[j for i in range(HEIGHT)] for j in range(WIDTH)])
YMAP = np.array([[i for i in range(HEIGHT)] for j in range(WIDTH)])

CAM_MAT = np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]])
CAM_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


