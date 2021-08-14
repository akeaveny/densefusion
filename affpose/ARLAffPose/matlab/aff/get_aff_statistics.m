function get_aff_statistics
close all; clc;clear;

opt = globals();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read class names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(opt.object_part_classes_file(), 'r');
C = textscan(fid, '%s');
classes = C{1};
% classes{end+1} = 'All Objects';
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results_keyframe = load('results_aff_keyframe.mat');
results_class_ids = results_keyframe.results_class_ids;
errors_add = results_keyframe.errors_add;
errors_add_s = results_keyframe.errors_add_s;
errors_rotation = results_keyframe.errors_rotation;
errors_translation = results_keyframe.errors_translation;
errors_translation_x = results_keyframe.errors_translation_x;
errors_translation_y = results_keyframe.errors_translation_y;
errors_translation_z = results_keyframe.errors_translation_z;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_distance = 0.1;
auc_threshold = 0.02;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting configs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for class_id = 1:numel(classes)
    
    index = find(results_class_ids == class_id);
    if isempty(index)
        index = 1:size(errors_add,1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ADD
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_add(index);
    D(D > max_distance) = inf;
    d = sort(D);
    n = numel(d);
    c = numel(d(d < auc_threshold));
    accuracy = cumsum(ones(1, n)) / n;
    AUC = VOCap(d, accuracy);
    fprintf('%30s, \tIndex:%d, \tAUC:%.2f, \tADD<2cm:%.2f,\n', char(classes(class_id)), length(index), AUC*100, (c/n)*100)
end

fprintf('\n\n')
for class_id = 1:numel(classes)
    
    index = find(results_class_ids == class_id);
    if isempty(index)
        index = 1:size(errors_add,1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ADD-S
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_add_s(index);
    D(D > max_distance) = inf;
    d = sort(D);
    n = numel(d);
    c = numel(d(d < auc_threshold));
    accuracy = cumsum(ones(1, n)) / n;
    AUC = VOCap(d, accuracy);
    fprintf('%30s, \tIndex:%d, \tAUC:%.2f, \tADD-S<2cm:%.2f,\n', char(classes(class_id)), length(index), AUC*100, (c/n)*100)
end

fprintf('\n\n')
for class_id = 1:numel(classes)
    
    index = find(results_class_ids == class_id);
    if isempty(index)
        index = 1:size(errors_add,1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ROTATIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_rotation(index);
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    fprintf('%30s, \tIndex:%d, \tMean Rotation Error:%.2f [deg]\n', char(classes(class_id)), length(index), mean(d))
    
end

fprintf('\n\n')
for class_id = 1:numel(classes)
    
    index = find(results_class_ids == class_id);
    if isempty(index)
        index = 1:size(errors_add,1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TRANSLATIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_translation(index);
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    fprintf('%30s, \tIndex:%d, \tMean Translation Error:%.2f [cm]\n', char(classes(class_id)), length(index), mean(d)*100)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TRANSLATIONS: X, Y, Z
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_translation_x(index);
%     fprintf('%20s, \tIndex:%d, \tMean X:%.2f [cm]\n', char(classes(class_id)), length(index), mean(D)*100)
    D = errors_translation_y(index);
%     fprintf('%20s, \tIndex:%d, \tMean Y:%.2f [cm]\n', char(classes(class_id)), length(index), mean(D)*100)
    D = errors_translation_z(index);
%     fprintf('%20s, \tIndex:%d, \tMean Z:%.2f [cm]\n', char(classes(class_id)), length(index), mean(D)*100)
    
end

function ap = VOCap(rec, prec)

try 
    index = isfinite(rec);
    rec = rec(index);
    prec = prec(index)';

    mrec=[0 ; rec ; 0.1];
    % disp(prec)
    % disp(prec(end))
    % disp(length(prec))
    % if length(prec) == 0
    %     prec(1) = 1;
    % end
    % disp(prec(end))

    mpre=[0 ; prec ; prec(end)];
    for i = 2:numel(mpre)
        mpre(i) = max(mpre(i), mpre(i-1));
    end
    i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
    ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) * 10;
catch
    ap = 0;
end