
import numpy as np

######################
# IMG UTILS
######################

def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def print_depth_info(depth):
    depth = np.array(depth)
    print("\tDepth of type:{} has min:{} & max:{}".format(depth.dtype, np.min(depth), np.max(depth)))

def print_class_labels(label):
    class_ids = np.unique(np.array(label, dtype=np.uint8))
    class_ids = class_ids[1:] # exclude the backgound
    print("\tMask has {} Labels: {}".format(len(class_ids), class_ids))