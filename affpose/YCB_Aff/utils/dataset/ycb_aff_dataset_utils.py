import glob
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import argparse

##################################
##################################

def sorted_cls_indexes(ycb_folder, obj_cls_idxs):
    sorted_cls_indexes = np.arange(start=0, stop=len(obj_cls_idxs))
    if ycb_folder == '0000':
        return np.array([1, 0, 2])
    elif ycb_folder == '0001':
        return np.array([1, 0, 2, 3]) # cracker box
    elif ycb_folder == '0002':
        # [ 5  7  9 10 16 18]
        return np.array([0, 5, 1, 2, 3, 4])
    elif ycb_folder == '0003':
        return np.array([2, 3, 0, 1])
    elif ycb_folder == '0004':
        return np.array([2, 3, 0, 1])
    elif ycb_folder == '0005':
        return np.array([1, 0, 2, 3])
    elif ycb_folder == '0006':
        return np.array([0, 4, 2, 1, 3])
    elif ycb_folder == '0007':
        return np.array([3, 2, 1, 0])
    elif ycb_folder == '0008':
        return np.array([2, 0, 1, 3, 4])
    elif ycb_folder == '0011':
        # [12 13 14 15 19 20]
        return np.array([5, 0, 1, 2, 3, 4])
    elif ycb_folder == '0013':
        # [ 1  4 10 13 17]
        return np.array([3, 0, 1, 2, 4])
    elif ycb_folder == '0014':
        # [ 1  3  9 11]
        return np.array([0, 1, 3, 2])

##################################
##################################

def map_obj_id_to_name(object_id):

    if object_id == 1:
        return 'master_chef_can'
    elif object_id == 2:
        return 'cracker_box'
    elif object_id == 3:
        return 'sugar_box'
    elif object_id == 4:
        return 'tomato_soup_can'
    elif object_id == 5:
        return 'mustard_bottle'
    elif object_id == 6:
        return 'tuna_fish_can'
    elif object_id == 7:
        return 'pudding_box'
    elif object_id == 8:
        return 'gelatin_box'
    elif object_id == 9:
        return 'potted_meat_can'
    elif object_id == 10:
        return 'banana'
    elif object_id == 11:
        return 'pitcher_base'
    elif object_id == 12:
        return 'bleach_cleanser'
    elif object_id == 13:
        return 'bowl'
    elif object_id == 14:
        return 'mug'
    elif object_id == 15:
        return 'power_drill'
    elif object_id == 16:
        return 'wood_block'
    elif object_id == 17:
        return 'scissors'
    elif object_id == 18:
        return 'large_marker'
    elif object_id == 19:
        return 'large_clamp'
    elif object_id == 20:
        return 'extra_large_clamp'
    elif object_id == 21:
        return 'foam_brick'
    else:
        print(" --- Object ID does not exist in UMD --- ")
        exit(1)

##################################
##################################

def map_obj_ids_to_obj_part_ids(object_id):

    if object_id == 1:          # 002_master_chef_can_16k
        return [1]
    elif object_id == 2:        # 003_cracker_box_16k
        return [2]
    elif object_id == 3:        # 004_sugar_box_16k
        return [3]
    elif object_id == 4:        # 005_tomato_soup_can_16k
        return [4]
    elif object_id == 5:        # 006_mustard_bottle_16k
        return [5]
    elif object_id == 6:        # 007_tuna_fish_can_16k
        return [6]
    elif object_id == 7:        # 008_pudding_box_16k
        return [7]
    elif object_id == 8:        # 009_gelatin_box_16k
        return [8]
    elif object_id == 9:        # 010_potted_meat_can_16k
        return [9]
    elif object_id == 10:       # 011_banana_16k
        return [10]
    if object_id == 11:         # 019_pitcher_base_16k
        return [11, 12, 13]
    elif object_id == 12:       # 021_bleach_cleanser_16k
        return [14]
    elif object_id == 13:       # 024_bowl_16k
        return [15, 16]
    elif object_id == 14:       # 025_mug_16k
        return [17, 18, 19]
    elif object_id == 15:       # 035_power_drill_16k
        return [20, 21, 22]
    elif object_id == 16:       # 036_wood_block_16k
        return [23]
    elif object_id == 17:       # 037_scissors_16k
        return [24, 25]
    elif object_id == 18:       # 040_large_marker_16k
        return [26]
    elif object_id == 19:       # 051_large_clamp_16k
        return [27, 28]
    elif object_id == 20:       # 052_extra_large_clamp_16k
        return [29, 30]
    elif object_id == 21:       # 061_foam_brick_16k
        return [31]
    else:
        print(" --- Object ID does not map to Object Parts --- ")
        exit(1)

def map_obj_part_ids_to_obj_id(obj_part_id):

    if obj_part_id in np.array([1]):            # 002_master_chef_can_16k
        return 1
    elif obj_part_id in np.array([2]):          # 003_cracker_box_16k
        return 2
    elif obj_part_id in np.array([3]):          # 004_sugar_box_16k
        return 3
    elif obj_part_id in np.array([4]):          # 005_tomato_soup_can_16k
        return 4
    elif obj_part_id in np.array([5]):          # 006_mustard_bottle_16k
        return 5
    elif obj_part_id in np.array([6]):          # 007_tuna_fish_can_16k
        return 6
    elif obj_part_id in np.array([7]):          # 008_pudding_box_16k
        return 7
    elif obj_part_id in np.array([8]):          # 009_gelatin_box_16k
        return 8
    elif obj_part_id in np.array([9]):          # 010_potted_meat_can_16k
        return 9
    elif obj_part_id in np.array([10]):         # 011_banana_16k
        return 10
    elif obj_part_id in np.array([11, 12, 13]): # 019_pitcher_base_16k
        return 11
    elif obj_part_id in np.array([14]):         # 021_bleach_cleanser_16k
        return 12
    elif obj_part_id in np.array([15, 16]):     # 024_bowl_16k
        return 13
    elif obj_part_id in np.array([17, 18, 19]): # 025_mug_16k
        return 14
    elif obj_part_id in np.array([20, 21, 22]): # 035_power_drill_16k
        return 15
    elif obj_part_id in np.array([23]):        # 036_wood_block_16k
        return 16
    elif obj_part_id in np.array([24, 25]):    # 037_scissors_16k
        return 17
    elif obj_part_id in np.array([26]):        # 040_large_marker_16k
        return 18
    elif obj_part_id in np.array([27, 28]):    # 051_large_clamp_16k
        return 19
    elif obj_part_id in np.array([29, 30]):    # 052_extra_large_clamp_16k
        return 20
    elif obj_part_id in np.array([31]):       # 061_foam_brick_16k
        return 21
    else:
        print(" --- Object ID does not map to Object Parts --- ")
        exit(1)

##################################
##################################

def map_obj_part_ids_to_aff_ids(aff_id):

    if aff_id in [11, 17, 20, 24]:                                              # grasp
        return 1
    elif aff_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 27, 29]:     # wrap-grasp
        return 2
    elif aff_id in [21, 23, 26, 31]:                                            # support
        return 3
    elif aff_id in [13, 16, 19]:                                                # contain
        return 4
    elif aff_id in [25]:                                                        # cut
        return 5
    elif aff_id in [28, 30]:                                                    # clamp
        return 6
    elif aff_id in [22]:                                                        # drill
        return 7
    else:
        print(" --- Object Part ID does not map to Affordance Label --- ")
        exit(1)

##################################
##################################

def colorize_obj_mask(instance_mask):

    instance_to_color = obj_color_map_dict()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def obj_color_map_dict():
    # https://htmlcolorcodes.com/
    # [red, blue, green]

    obj_color_map_dict = {
        0: [0, 0, 0],
        1: [255, 73, 51],    # red 1
        2: [255, 113, 51],   # orange 1
        3: [235, 195, 17],   # gold
        4: [255, 202, 51],   # yellow 1
        5: [255, 240, 51],   # yellow 2
        6: [209, 255, 51],   # light green 1
        7: [125, 255, 51],   # light green 2
        8: [51,  255, 91],   # neon green
        9: [51,  255, 175],  # teal 1
        10: [51, 255, 233],  # teal 2
        11: [51, 199, 233],  # light blue 1
        12: [51, 206, 255],  # light blue 2
        13: [51, 138, 255],  # blue 1
        14: [51, 76,  255],  # blue 2
        15: [113, 51, 255],  # purple 1
        16: [168, 51, 255],  # purple 2
        17: [224, 51, 255],  # pink 1
        18: [255, 51, 252],  # hot pink
        19: [255, 51, 196],  # dark pink 1
        20: [51, 255, 233],  # orange 2
        21: [255, 51, 138],  # dark pink 2
    }

    return obj_color_map_dict

##################################
##################################

def densefusion_pred_color():
    return (150, 247, 241) # teal

def pose_cnn_pred_color():
    return (247, 163, 150) # teal

def obj_color_map(idx):
    ''' [red, blue, green]'''

    if idx == 1:
        return (255, 73, 51)
    elif idx == 2:
        return (255, 113, 51)
    elif idx == 3:
        return (235, 195, 17)
    elif idx == 4:
        return (255, 202, 51)
    elif idx == 5:
        return (255, 240, 51)
    elif idx == 6:
        return (209, 255, 51)
    elif idx == 7:
        return (125, 255, 51)
    elif idx == 8:
        return (51,  255, 91)
    elif idx == 9:
        return (51,  255, 175)
    elif idx == 10:
        return (51, 255, 233)
    elif idx == 11:
        return (51, 199, 233)
    elif idx == 12:
        return (51, 206, 255)
    elif idx == 13:
        return (51, 138, 255)
    elif idx == 14:
        return (51, 76,  255)
    elif idx == 15:
        return (113, 51, 255)
    elif idx == 16:
        return (168, 51, 255)
    elif idx == 17:
        return (224, 51, 255)
    elif idx == 18:
        return (255, 51, 252)
    elif idx == 19:
        return (255, 51, 196)
    elif idx == 20:
        return (51, 255, 233)
    elif idx == 21:
        return (255, 51, 138)
    else:
        print(" --- idx does not map to a colour --- ")
        exit(1)

##################################
##################################

def colorize_aff_mask(instance_mask):

    instance_to_color = aff_color_map_dict()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def aff_color_map_dict():
    ''' [red, blue, green]'''

    aff_color_map_dict = {
        0: [0, 0, 0],
        1: [133, 17, 235],  # red
        2: [235, 96, 17],   # orange
        3: [235, 195, 17],  # gold
        4: [176, 235, 17],  # light green/yellow
        5: [76, 235, 17],   # green
        6: [17, 235, 139],  # teal
        7: [17, 235, 225],  # light blue
    }

    return aff_color_map_dict

##################################
##################################

def aff_color_map(idx):
    ''' [red, blue, green]'''

    if idx == 1:
        return (133,  17, 235)        # red
    elif idx == 2:
        return (235, 96, 17)        # orange
    elif idx == 3:
        return (235, 195, 17)       # gold
    elif idx == 4:
        return (176,  235, 17)      # light green/yellow
    elif idx == 5:
        return (76,   235, 17)      # green
    elif idx == 6:
        return (17,  235, 139)      # teal
    elif idx == 7:
        return (17,  235, 225)      # light blue
    else:
        print(" --- idx does not map to a colour --- ")
        exit(1)