import numpy as np

#######################################
#######################################

import affpose.YCB_Aff.cfg as config

from affpose.YCB_Aff.utils import helper_utils
from affpose.YCB_Aff.utils.dataset import ycb_aff_dataset_utils

#######################################
#######################################

def load_obj_part_ply_files():

    ###################################
    # YCB CONFIG
    ###################################

    classes_file = open(config.CLASSES_FILE)
    class_ids_file = open(config.CLASS_IDS_FILE)
    class_ids = np.loadtxt(class_ids_file, dtype=np.int32)

    ###################################
    # OBJECT
    ###################################

    cld = {}
    for class_id in class_ids:
        class_input = classes_file.readline()
        if not class_input:
            break
        input_file = open(config.DATASET_ROOT_PATH + 'models/{0}/points.xyz'.format(class_input[:-1]))
        cld[class_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld[class_id] = np.array(cld[class_id])
        # print("class_id: ", class_id)
        # print("class_input: ", class_input.rstrip())
        # print("Num Point Clouds: {}\n".format(len(cld[class_id])))
        input_file.close()

    ###################################
    # OBJECT CENTERED
    ###################################

    class_obj_part_file = open(config.OBJ_PART_CLASSES_FILE)
    class_obj_part_id_file = open(config.OBJ_PART_CLASS_IDS_FILE)
    class_obj_part_IDs = np.loadtxt(class_obj_part_id_file, dtype=np.int32)

    cld_obj_centered = {}
    for class_obj_part_id in class_obj_part_IDs:
        class_input = class_obj_part_file.readline()
        if not class_input:
            break
        input_file = open(config.AFF_DATASET_ROOT_PATH + 'ycb_affordance_models/{0}/densefusion/{0}_obj_centered.xyz'.format(class_input[:-1]))
        cld_obj_centered[class_obj_part_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld_obj_centered[class_obj_part_id].append(
                [float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld_obj_centered[class_obj_part_id] = np.array(cld_obj_centered[class_obj_part_id])
        # print("class_id: ", class_obj_part_id)
        # print("class_input: ", class_input.rstrip())
        # print("Num Point Clouds: {}\n".format(len(cld_obj_centered[class_obj_part_id])))
        input_file.close()

    ###################################
    # OBJECT PART CENTERED
    ###################################

    class_obj_part_file = open(config.OBJ_PART_CLASSES_FILE)
    class_obj_part_id_file = open(config.OBJ_PART_CLASS_IDS_FILE)
    class_obj_part_IDs = np.loadtxt(class_obj_part_id_file, dtype=np.int32)

    cld_obj_part_centered = {}
    for class_obj_part_id in class_obj_part_IDs:
        class_input = class_obj_part_file.readline()
        if not class_input:
            break
        input_file = open(config.AFF_DATASET_ROOT_PATH + 'ycb_affordance_models/{0}/densefusion/{0}_aff_centered.xyz'.format(class_input[:-1]))
        cld_obj_part_centered[class_obj_part_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld_obj_part_centered[class_obj_part_id].append(
                [float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld_obj_part_centered[class_obj_part_id] = np.array(cld_obj_part_centered[class_obj_part_id])
        # print("class_id: ", class_obj_part_id)
        # print("class_input: ", class_input.rstrip())
        # print("Num Point Clouds: {}\n".format(len(cld_obj_centered[class_obj_part_id])))
        input_file.close()

    ##################################
    # CHECKING OBJECT PART LABELS
    ##################################

    class_file = open(config.CLASSES_FILE)
    class_obj_part_file = open(config.OBJ_PART_CLASSES_FILE)
    obj_classes = np.loadtxt(class_file, dtype=np.str)
    obj_part_classes = np.loadtxt(class_obj_part_file, dtype=np.str)

    class_id_file = open(config.CLASS_IDS_FILE)
    class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
    class_obj_part_id_file = open(config.OBJ_PART_CLASS_IDS_FILE)
    class_obj_part_IDs = np.loadtxt(class_obj_part_id_file, dtype=np.int32)

    for class_ID in class_IDs:
        print("\n*** Mapping Object: ID:{}, Name: {}, cld:{} ***".format(class_ID, obj_classes[int(class_ID) - 1], len(cld[class_ID])))
        obj_part_ids = ycb_aff_dataset_utils.map_obj_ids_to_obj_part_ids(class_ID)
        for obj_part_id in obj_part_ids:
            print("\tObject Part: ID:{}, Name: {}, cld:{}".format(obj_part_id, obj_part_classes[int(obj_part_id) - 1], len(cld_obj_part_centered[obj_part_id])))
    print("")

    ##################################
    # TRAIN
    ##################################

    class_obj_part_file = open(config.OBJ_PART_CLASSES_FILE_TRAIN)
    obj_part_classes = np.loadtxt(class_obj_part_file, dtype=np.str)

    class_obj_part_id_file = open(config.OBJ_PART_CLASS_IDS_FILE_TRAIN)
    class_obj_part_IDs = np.loadtxt(class_obj_part_id_file, dtype=np.int32)

    ##################################
    ##################################

    return cld, cld_obj_centered, cld_obj_part_centered, \
           obj_classes, obj_part_classes, class_IDs, class_obj_part_IDs