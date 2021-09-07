function plot_accuracy_keyframe
clc;clear;

opt = globals();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read class names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(opt.classes_file(), 'r');
C = textscan(fid, '%s');
classes = C{1};
classes{end+1} = 'All 21 objects';
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
object = load('results_obj_keyframe.mat');
results_class_ids = object.results_class_ids;
errors_add = object.errors_add;
errors_add_s = object.errors_add_s;
errors_rotation = object.errors_rotation;
errors_translation = object.errors_translation;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting configs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_distance = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting configs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for class_id = 1:numel(classes)
%     fprintf('eval for %s\n', char(classes(class_id)))
    
    index = find(results_class_ids == class_id);
%     disp(length(index));
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
    c = numel(d(d < 0.02));
    accuracy = cumsum(ones(1, n)) / n;
    AUC = VOCap(d, accuracy);
    fprintf('%20s,\tIndex:%d\tAUC:%.2f,\tADD-S<2cm:%.2f,\n', char(classes(class_id)), length(index), AUC*100, (c/n)*100)
    
end

function ap = VOCap(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';

mrec=[0 ; rec ; 0.1];
% disp(prec)
% disp(end)
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