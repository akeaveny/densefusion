function evaluate_poses_keyframe

clear;clc;

opt = globals();
delete 'results/results_keyframe.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read class names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(opt.classes_file(), 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read model names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_objects = numel(object_names);
models = cell(num_objects, 1);
for i = 1:num_objects
    filename = fullfile(opt.dataset_root, 'object_meshes/models', object_names{i}, 'densefusion', strcat(object_names{i}, '.xyz'));
    disp(filename);
    models{i} = load(filename);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read keyframes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gt_keyframes = dir(fullfile(opt.eval_folder_gt, '*.mat'));
df_wo_refine_keyframes = dir(fullfile(opt.eval_folder_df_wo_refine, '*.mat'));
df_iterative_keyframes = dir(fullfile(opt.eval_folder_df_iterative, '*.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_preds = length(gt_keyframes)*6; % ~6 objects per scene

results_class_ids  = zeros(num_preds, 1);
errors_add         = zeros(num_preds, 1);
errors_add_s       = zeros(num_preds, 1);
errors_rotation    = zeros(num_preds, 1); 
errors_translation = zeros(num_preds, 1);

count = 0;
for i = 1:numel(gt_keyframes)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load gt poses
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = strcat(gt_keyframes(i).folder, '/' , gt_keyframes(i).name);
    gt = load(filename);
    % display filename
    disp(filename);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load gt poses
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = strcat(df_iterative_keyframes(i).folder, '/' , df_iterative_keyframes(i).name);
    pred = load(filename);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % for each class
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:numel(gt.class_ids)
        % class id         
        count = count + 1;
        cls_index = gt.class_ids(j);
        results_class_ids(count) = cls_index;
        
        % Load gt & pred poses
        gt_pose = gt.poses(:, :, j);
        GT(1:3, 1:3) = quat2rotm(gt_pose(1:4));
        GT(:, 4) = gt_pose(5:7);
            
        pred_pose = pred.poses(:, :, j);
        PRED(1:3, 1:3) = quat2rotm(pred_pose(1:4));
        PRED(:, 4) = pred_pose(5:7);
        
        % error metrics
        pointcloud = models{cls_index}(:, 1:3); % remove colour from xyz
        errors_add(count)         = add(PRED, GT, pointcloud');
        errors_add_s(count)       = adi(PRED, GT, pointcloud');
        errors_rotation(count)    = re(PRED(1:3, 1:3), GT(1:3, 1:3));
        errors_translation(count) = te(PRED(:, 4), GT(:, 4));        
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('results/results_keyframe.mat', ...
'results_class_ids',...
'errors_add', ...
'errors_add_s',...
'errors_rotation',...
'errors_translation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pts_new = transform_pts_Rt(pts, RT)
%     """
%     Applies a rigid transformation to 3D points.
% 
%     :param pts: nx3 ndarray with 3D points.
%     :param R: 3x3 rotation matrix.
%     :param t: 3x1 translation vector.
%     :return: nx3 ndarray with transformed 3D points.
%     """
n = size(pts, 2);
pts_new = RT * [pts; ones(1, n)];

function error = add(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with no indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);
diff = pts_est - pts_gt;
error = mean(sqrt(sum(diff.^2, 1)));

function error = adi(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);

% Calculate distances to the nearest neighbors from pts_gt to pts_est
MdlKDT = KDTreeSearcher(pts_est');
[~, D] = knnsearch(MdlKDT, pts_gt');
error = mean(D);

function error = re(R_est, R_gt)
%     """
%     Rotational Error.
% 
%     :param R_est: Rotational element of the estimated pose (3x1 vector).
%     :param R_gt: Rotational element of the ground truth pose (3x1 vector).
%     :return: Error of t_est w.r.t. t_gt.
%     """

error_cos = 0.5 * (trace(R_est * inv(R_gt)) - 1.0);
error_cos = min(1.0, max(-1.0, error_cos));
error = acos(error_cos);
error = 180.0 * error / pi;

function error = te(t_est, t_gt)
% """
% Translational Error.
% 
% :param t_est: Translation element of the estimated pose (3x1 vector).
% :param t_gt: Translation element of the ground truth pose (3x1 vector).
% :return: Error of t_est w.r.t. t_gt.
% """
error = norm(t_gt - t_est);