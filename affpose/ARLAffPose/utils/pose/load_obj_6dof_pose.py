
import yaml
import numpy as np

from scipy.spatial.transform import Rotation as R

#######################################
#######################################

def load_obj_6dof_pose(yaml_addr):
    #####################
    #####################
    yaml_file = open(yaml_addr, 'r')
    parsed = yaml.load(yaml_file, Loader=yaml.FullLoader)

    _obj_ids = []
    _obj_poses = []
    for idx, obj in enumerate(parsed.keys()):
        label = np.asarray(parsed[obj]['label'], dtype=np.uint8)
        _obj_ids.append(label)
        # translation
        trans = parsed[obj]['pose'][0]
        # rotation
        quart = parsed[obj]['pose'][1]  # x y z w
        quart.append(quart[0])
        quart.pop(0)
        rot = R.from_quat(quart)  # x y z w
        pose = rot.as_matrix().tolist()

        for i in range(0, 3):
            pose[i].append(trans[i])

        if idx == 0:
            for i in range(0, 3):
                row = []
                for k in range(0, 4):
                    ele = []
                    ele.append(pose[i][k])
                    row.append(ele)
                _obj_poses.append(row)
        else:
            for i in range(0, 3):
                for k in range(0, 4):
                    _obj_poses[i][k].append(pose[i][k])

    obj_ids = np.asarray(_obj_ids, dtype=np.uint8)

    obj_poses = np.asarray(_obj_poses)
    obj_poses = np.reshape(obj_poses, (3, 4, len(parsed)))

    return obj_ids, obj_poses