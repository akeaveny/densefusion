
import yaml
import numpy as np


#######################################
#######################################

def load_camera_pose(posegraph_addr):
    return np.loadtxt(posegraph_addr, dtype=np.float32)[:, 1:] # we exclude the timestamp