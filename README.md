# DenseFusion

This work forks [DenseFusion](https://github.com/j96w/DenseFusion), to work with a custom dataset I created. Here is the link to the[ original paper for DenseFusion](https://arxiv.org/abs/1901.04780). 

I used densefusion with the following repos:

1. [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) for generating real images.
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating synthetic images.
3. [arl-affpose-dataset-utils](https://github.com/UW-Advanced-Robotics-Lab/arl-affpose-dataset-utils) a custom dataset that I generated.
4. [pytorch-simple-affnet](https://github.com/UW-Advanced-Robotics-Lab/pytorch-simple-affnet) for predicting an object affordance labels.
6. [arl-affpose-ros-node](https://github.com/UW-Advanced-Robotics-Lab/arl-affpose-ros-node): for deploying our network for 6-DoF pose estimation with our ZED camera.
7. [barrett_tf_publisher](https://github.com/UW-Advanced-Robotics-Lab/barrett-wam-arm) for robotic grasping experiments. Specifically barrett_tf_publisher and barrett_trac_ik. 

In the sample below we see real time implementation on our 7-DoF Robot Arm.
![Alt text](samples/demo.gif?raw=true "Title")

## Requirements
   ```
   conda env create -f environment.yml --name DenseFusion
   ```

## AffNet
1. To inspect ground truth object pose (first look at relative paths for root folder of dataset in tools/ARLAffPose/cfg.py):
   ```
   python tools/ARLAffPose/scripts/load_gt_obj_poses.py
   ```
1. To inspect ground truth affordance pose (see [PyTorch-Simple-AffNet](https://github.com/akeaveny/PyTorch-Simple-AffNet/blob/master/README.md)):
   ```
   python tools/ARLAffPose/scripts/load_gt_obj_part_poses.py
   ```
3. To run training:
   ```
   python tools/train.py
   ```
4. To get predicted pose run:
   ```
   python tools/ARLAffPose/scripts/evaluate_poses_keyframe.py
   ```
