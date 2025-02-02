import numpy as np

#######################################
#######################################

ROOT_PATH = '/home/akeaveny/git/DenseFusion/'

# /home/akeaveny/git/DenseFusion/datasets/ycb_aff/
CLASSES_FILE   = ROOT_PATH + 'datasets/ycb_aff/dataset_config/classes.txt'
CLASS_IDS_FILE = ROOT_PATH + 'datasets/ycb_aff/dataset_config/class_ids.txt'
OBJ_PART_CLASSES_FILE   = ROOT_PATH + 'datasets/ycb_aff/dataset_config/obj_part_classes.txt'
OBJ_PART_CLASS_IDS_FILE = ROOT_PATH + 'datasets/ycb_aff/dataset_config/obj_part_classes_ids.txt'
OBJ_PART_CLASSES_FILE_TRAIN   = ROOT_PATH + 'datasets/ycb_aff/dataset_config/obj_part_classes_train.txt'
OBJ_PART_CLASS_IDS_FILE_TRAIN = ROOT_PATH + 'datasets/ycb_aff/dataset_config/obj_part_classes_ids_train.txt'

TRAIN_FILE = ROOT_PATH + 'datasets/ycb_aff/dataset_config/train_data_list.txt'
TEST_FILE  = ROOT_PATH + 'datasets/ycb_aff/dataset_config/test_data_list.txt'

# Trained models
DF_GITHUB_TRAINED_MODEL = '/data/Akeaveny/weights/DenseFusion/ycb/densefusion/pose_model_26_0.012863246640872631.pth'
DF_GITHUB_TRAINED_REFINE_MODEL = '/data/Akeaveny/weights/DenseFusion/ycb/densefusion/pose_refine_model_69_0.009449292959118935.pth'
TRAINED_MODEL = '/data/Akeaveny/weights/DenseFusion/ycb/real_and_syn/pose_model_27_0.012961520093793814.pth'
TRAINED_REFINE_MODEL = '/data/Akeaveny/weights/DenseFusion/ycb/real_and_syn/pose_refine_model_93_0.009422253060541326.pth'
TRAINED_AFF_MODEL = '/data/Akeaveny/weights/DenseFusion/ycb_aff/real_and_syn/pose_model_20_0.012745570227629784.pth'
TRAINED_AFF_REFINE_MODEL = '/data/Akeaveny/weights/DenseFusion/ycb_aff/real_and_syn/pose_refine_model_57_0.009359799028407731.pth'

# MATLAB
OBJ_EVAL_FOLDER_GT = ROOT_PATH + 'affpose/YCB_Aff/matlab/obj/results/gt'
OBJ_EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'affpose/YCB_Aff/matlab/obj/results/df_wo_refine'
OBJ_EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'affpose/YCB_Aff/matlab/obj/results/df_iterative'
AFF_EVAL_FOLDER_GT = ROOT_PATH + 'affpose/YCB_Aff/matlab/aff/results/gt'
AFF_EVAL_FOLDER_DF_WO_REFINE = ROOT_PATH + 'affpose/YCB_Aff/matlab/aff/results/df_wo_refine'
AFF_EVAL_FOLDER_DF_ITERATIVE = ROOT_PATH + 'affpose/YCB_Aff/matlab/aff/results/df_iterative'

#######################################
### YCB
#######################################

DATASET_ROOT_PATH = '/data/Akeaveny/Datasets/YCB_Video_Dataset/'
AFF_DATASET_ROOT_PATH = '/data/Akeaveny/Datasets/YCB_Affordance_Dataset/'

YCB_TOOLBOX_CONFIG = ROOT_PATH + 'YCB_Video_toolbox/results_PoseCNN_RSS2018/'

RGB_EXT = '-color.png'
DEPTH_EXT = '-depth.png'
LABEL_EXT = '-label.png'
META_EXT = '-meta.mat'
BOX_EXT = '-box.txt'

OBJ_PART_LABEL_EXT = '-obj_part_label.png'
AFF_LABEL_EXT = '-aff_label.png'

POSECNN_EXT = '.mat'

#######################################
### YCB AFF
#######################################

NUM_OBJECTS = 21
NUM_OBJECTS_PARTS = 31

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

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

NUM_PT = 1000
NUM_PT_MESH_SMALL = 500
NUM_PT_MESH_LARGE = 2600

FRONT_NUM = 2

REFINE_ITERATIONS = 2
BATCH_SIZE = 1

PRED_C_THRESHOLD = 0.0



