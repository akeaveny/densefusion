function opt = globals()

opt.root = '/home/akeaveny/git/DenseFusion/';
opt.dataset_root = '/data/Akeaveny/Datasets/YCB_Affordance_Dataset/';

opt.classes_file = strcat(opt.root(), 'datasets/ycb_aff/dataset_config/obj_part_classes_train.txt');

opt.keyframes = strcat(opt.root(), 'datasets/ycb/dataset_config/keyframe.txt');

opt.posecnn_results = strcat(opt.root(), 'YCB_Video_toolbox/results_PoseCNN_RSS2018/');

opt.obj_eval_folder_gt           = strcat(opt.root(), 'affpose/YCB_Aff/matlab/obj/results/gt/');
opt.obj_eval_folder_df_wo_refine = strcat(opt.root(), 'affpose/YCB_Aff/matlab/obj/results/df_wo_refine/');
opt.obj_eval_folder_df_iterative = strcat(opt.root(), 'affpose/YCB_Aff/matlab/obj/results/df_iterative/');

opt.aff_eval_folder_gt           = strcat(opt.root(), 'affpose/YCB_Aff/matlab/aff/results/gt/');
opt.aff_eval_folder_df_wo_refine = strcat(opt.root(), 'affpose/YCB_Aff/matlab/aff/results/df_wo_refine/');
opt.aff_eval_folder_df_iterative = strcat(opt.root(), 'affpose/YCB_Aff/matlab/aff/results/df_iterative/');