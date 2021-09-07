import numpy as np

#######################################
#######################################

from affpose.YCB import cfg as config

from affpose.YCB.utils import helper_utils
from affpose.YCB.utils.dataset import ycb_dataset_utils

#######################################
#######################################

def load_obj_ply_files():

    ###################################
    # YCB CONFIG
    ###################################

    classes_file = open(config.CLASSES_FILE)
    class_ids_file = open(config.CLASS_IDS_FILE)
    class_ids = np.loadtxt(class_ids_file, dtype=np.int32)

    ###################################
    ###################################
    print()

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
        print("class_id: ", class_id)
        print("class_input: ", class_input.rstrip())
        print("Num Point Clouds: {}\n".format(len(cld[class_id])))
        input_file.close()

    ##################################
    # CHECKING AFF LABELS
    ##################################

    classes_file = open(config.CLASSES_FILE)
    class_ids_file = open(config.CLASS_IDS_FILE)

    obj_classes = np.loadtxt(classes_file, dtype=np.str)
    obj_class_ids = np.loadtxt(class_ids_file, dtype=np.int32)

    ##################################
    ##################################

    return cld, obj_classes, obj_class_ids