function opt = globals()

opt.root = '/home/akeaveny/git/DenseFusion/';
opt.dataset_root = '/data/Akeaveny/Datasets/YCB_Video_Dataset/';

opt.classes_file = strcat(opt.root(), 'datasets/ycb/dataset_config/classes.txt');


opt.keyframes = strcat(opt.root(), 'datasets/ycb/dataset_config/keyframe.txt');

opt.posecnn_results = strcat(opt.root(), 'YCB_Video_toolbox/results_PoseCNN_RSS2018/');

opt.eval_folder_gt           = strcat(opt.root(), 'affpose/YCB/matlab/results/gt/');
opt.eval_folder_posecnn      = strcat(opt.root(), 'affpose/YCB/matlab/results/posecnn/');
opt.eval_folder_df_wo_refine = strcat(opt.root(), 'affpose/YCB/matlab/results/df_wo_refine/');
opt.eval_folder_df_iterative = strcat(opt.root(), 'affpose/YCB/matlab/results/df_iterative/');