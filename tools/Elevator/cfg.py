import numpy as np

#######################################
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
### YCB
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/Elevator/'

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

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

CAMERA_SCALE = 1000
CAM_CX = 341.276
CAM_CY = 175.296
CAM_FX = 338.546630859375
CAM_FY = 338.546630859375

XMAP = np.array([[j for i in range(HEIGHT)] for j in range(WIDTH)])
YMAP = np.array([[i for i in range(HEIGHT)] for j in range(WIDTH)])

CAM_MAT = np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]])
CAM_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#######################################
# LabelFusion Log DIR
#######################################

LOG_FIlE = 'elavator_down_1/images/'
# LOG_FIlE = 'elavator_robohub_1/images/'

#######################################
#######################################

LABELFUSION_LOG_PATH = ROOT_DATA_PATH + 'LabelFusion/logs/' + LOG_FIlE
LABELFUSION_DATASET_PATH = ROOT_DATA_PATH + 'LabelFusion/dataset/' + LOG_FIlE


