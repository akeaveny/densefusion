import math
import numpy as np
from PIL import Image

######################
# IMG UTILS
######################

def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def print_depth_info(depth):
    depth = np.array(depth)
    print("Depth of type:{} has min:{} & max:{}".format(depth.dtype, np.min(depth), np.max(depth)))

def print_class_labels(label):
    class_ids = np.unique(np.array(label, dtype=np.uint8))
    class_ids = class_ids[1:] # exclude the backgound
    print("Mask has {} Labels: {}".format(len(class_ids), class_ids))

def crop(pil_img, crop_size, is_img=False):
    _dtype = np.array(pil_img).dtype
    pil_img = Image.fromarray(pil_img)
    crop_w, crop_h = crop_size
    img_width, img_height = pil_img.size
    left, right = (img_width - crop_w) / 2, (img_width + crop_w) / 2
    top, bottom = (img_height - crop_h) / 2, (img_height + crop_h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    # pil_img = pil_img.crop((left, top, right, bottom)).resize((crop_w, crop_h))
    pil_img = pil_img.crop((left, top, right, bottom))
    ###
    if is_img:
        img_channels = np.array(pil_img).shape[-1]
        img_channels = 3 if img_channels == 4 else img_channels
        resize_img = np.zeros((crop_h, crop_w, img_channels))
        resize_img[0:(bottom - top), 0:(right - left), :img_channels] = np.array(pil_img)[..., :img_channels]
    else:
        resize_img = np.zeros((crop_h, crop_w))
        resize_img[0:(bottom - top), 0:(right - left)] = np.array(pil_img)

    return np.array(resize_img, dtype=_dtype)

######################
# 3D UTILS
######################

def sort_imgpts(_imgpts):
    imgpts = np.squeeze(_imgpts.copy())
    # imgpts = imgpts[np.lexsort(np.transpose(imgpts)[::-1])]
    return np.int32([imgpts])