# DenseFusion

This work forks [DenseFusion](https://github.com/j96w/DenseFusion), to work with a custom dataset I created. Here is the link to the[ original paper for DenseFusion](https://arxiv.org/abs/1901.04780). 

I use DenseFusion with the following repos:

1. [Labelusion](https://github.com/akeaveny/LabelFusion) for generating Real Images
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating Synthetic Images
3. [PyTorch-Simple-AffNet](https://github.com/akeaveny/DenseFusion) for predicting 6-DoF Object Pose.
4. AffDenseFusionROSNode: coming soon.
5. [Barrett_WAM_Arm_tf_publisher](https://github.com/akeaveny/Barrett_WAM_Arm_tf_publisher) for transforming object pose reference frame to the base link of our 7-DoF manipulator.

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
