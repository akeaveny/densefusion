function evaluate_aff_poses_keyframe

opt = globals();
delete 'results_aff_keyframe.mat'

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
    filename = fullfile(opt.dataset_root, 'ycb_affordance_models/', object_names{i}, 'densefusion/', strcat(object_names{i}, '_aff_centered.xyz'));
    disp(filename);
    models{i} = load(filename);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read class names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(opt.keyframes(), 'r');
C = textscan(fid, '%s');
keyframes = C{1};
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gt_keyframes = dir(fullfile(opt.aff_eval_folder_gt, '*.mat'));
df_wo_refine_keyframes = dir(fullfile(opt.aff_eval_folder_df_wo_refine, '*.mat'));
df_iterative_keyframes = dir(fullfile(opt.aff_eval_folder_df_iterative, '*.mat'));
fprintf('Loaded %d Keyframes \n', numel(gt_keyframes))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results_class_ids  = zeros(15000, 1);

errors_add         = zeros(15000, 1);
errors_add_s       = zeros(15000, 1);
errors_rotation    = zeros(15000, 1); 
errors_translation = zeros(15000, 1);

count = 0;
for i = 1:numel(keyframes)
    
%     i = 2358 + i;
    
    % parse keyframe name
    name = keyframes{i};
    pos = strfind(name, '/');
    seq_id = str2double(name(1:pos-1));
    frame_id = str2double(name(pos+1:end));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load pred poses
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = strcat(df_iterative_keyframes(i).folder, '/' , df_iterative_keyframes(i).name);
    pred_results = load(filename);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load gt poses
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     filename = fullfile(opt.dataset_root(), 'data', sprintf('%04d/%06d-meta.mat', seq_id, frame_id));
    filename = strcat(gt_keyframes(i).folder, '/' , gt_keyframes(i).name);
    gt = load(filename);
    disp(filename);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % for each class
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     disp(gt.class_ids);
    for j = 1:numel(gt.class_ids)
        count = count + 1;
        
        cls_index = gt.class_ids(j);
        results_class_ids(count) = cls_index;
        
        % GT
%         RT_gt = gt.poses(:, :, j);
        RT_gt(1:3, 1:3) = quat2rotm(gt.poses(j, 1:4));
        RT_gt(:, 4) = gt.poses(j, 5:7);
        
        % network result
        roi_index = find(pred_results.class_ids == cls_index);
        if isempty(roi_index) == 0           
            
            % pose after ICP refinement
            RT(1:3, 1:3) = quat2rotm(pred_results.poses(roi_index,1:4));
            RT(:, 4) = pred_results.poses(roi_index, 5:7);
            
            pointcloud = models{cls_index}(:, 1:3); % remove colour from xyz
            
            errors_add(count) = add(RT, RT_gt, pointcloud');
            errors_add_s(count) = adi(RT, RT_gt, pointcloud');
            errors_rotation(count) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            errors_translation(count) = te(RT(:, 4), RT_gt(:, 4));
        
        else
            errors_add(count) = inf;
            errors_add_s(count) = inf;
            errors_rotation(count) = inf;
            errors_translation(count) = inf;
        end
        
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('results_aff_keyframe.mat', ...
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