function opt = globals()

opt.root = '/home/akeaveny/git/DenseFusion/';
opt.dataset_root = '/data/Akeaveny/Datasets/ARLAffPose/';

opt.classes_file = strcat(opt.root(), 'datasets/arl_affpose/dataset_config/classes.txt');
opt.object_part_classes_file = strcat(opt.root(), 'datasets/arl_affpose/dataset_config/obj_part_classes.txt');

opt.keyframes = strcat(opt.root(), 'datasets/arl_affpose/dataset_config/data_lists/test_list.txt');

opt.obj_eval_folder_gt           = strcat(opt.root(), 'affpose/ARLAffPose/matlab/obj/results/gt/');
opt.obj_eval_folder_df_wo_refine = strcat(opt.root(), 'affpose/ARLAffPose/matlab/obj/results/df_wo_refine/');
opt.obj_eval_folder_df_iterative = strcat(opt.root(), 'affpose/ARLAffPose/matlab/obj/results/df_iterative/');

opt.aff_eval_folder_gt           = strcat(opt.root(), 'affpose/ARLAffPose/matlab/aff/results/gt/');
opt.aff_eval_folder_df_wo_refine = strcat(opt.root(), 'affpose/ARLAffPose/matlab/aff/results/df_wo_refine/');
opt.aff_eval_folder_df_iterative = strcat(opt.root(), 'affpose/ARLAffPose/matlab/aff/results/df_iterative/');