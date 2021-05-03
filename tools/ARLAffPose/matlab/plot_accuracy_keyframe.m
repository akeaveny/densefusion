function plot_accuracy_keyframe
close all; clc;clear;

opt = globals();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read class names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(opt.classes_file(), 'r');
C = textscan(fid, '%s');
classes = C{1};
% classes{end+1} = 'All Objects';
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results_keyframe = load('results/results_keyframe.mat');
results_class_ids = results_keyframe.results_class_ids;
errors_add = results_keyframe.errors_add;
errors_add_s = results_keyframe.errors_add_s;
errors_rotation = results_keyframe.errors_rotation;
errors_translation = results_keyframe.errors_translation;
errors_translation_x = results_keyframe.errors_translation_x;
errors_translation_y = results_keyframe.errors_translation_y;
errors_translation_z = results_keyframe.errors_translation_z;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting configs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hf = figure('units','normalized','outerposition',[0 0 1 1]);
font_size = 12;
max_distance = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting configs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for class_id = 1:numel(classes)
    
    index = find(results_class_ids == class_id);
    if isempty(index)
        index = 1:size(errors_add,1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ADD-S
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_add(index);
    D(D > max_distance) = inf;
    d = sort(D);
    n = numel(d);
    c = numel(d(d < 0.02));
    accuracy = cumsum(ones(1, n)) / n;
    AUC = VOCap(d, accuracy);
    fprintf('%20s, \tIndex:%d, \tAUC:%.2f, \tADD<2cm:%.2f,\n', char(classes(class_id)), length(index), AUC*100, (c/n)*100)
    
    % plotting
    subplot(2, 2, 1);
    plot(d, accuracy,  'c', 'LineWidth', 4);
    legend_ = sprintf('ADD (AUC:%.2f)(<2cm:%.2f)', AUC*100, (c/n)*100);
    h = legend(legend_, 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Average distance threshold in meter (symmetry)');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(class_id, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);    
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    hold on;
    
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
    fprintf('%20s, \tIndex:%d, \tAUC:%.2f, \tADD-S<2cm:%.2f,\n', char(classes(class_id)), length(index), AUC*100, (c/n)*100)
    
    % plotting
    subplot(2, 2, 2);
    plot(d, accuracy,  'b', 'LineWidth', 4);
    legend_ = sprintf('ADD-S (AUC:%.2f)(<2cm:%.2f)', AUC*100, (c/n)*100);
    h = legend(legend_, 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Average distance threshold in meter (symmetry)');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(class_id, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);    
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    hold on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ROTATIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_rotation(index);
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    fprintf('%20s, \tIndex:%d, \tMean Rotation Error:%.2f [deg]\n', char(classes(class_id)), length(index), mean(d))
    
    % plotting
    subplot(2, 2, 3);
    plot(d, accuracy,  'r', 'LineWidth', 4);
    xline(mean(d),  'r', '--');
    set(h, 'FontSize', font_size);
    h = xlabel('Rotation angle threshold');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(class_id, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);    
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    hold on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TRANSLATIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_translation(index);
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    fprintf('%20s, \tIndex:%d, \tMean Translation Error:%.2f [cm]\n', char(classes(class_id)), length(index), mean(d)*100)
    
    % plotting
    subplot(2, 2, 4);
    plot(d, accuracy,  'm', 'LineWidth', 4);
    xline(mean(d),  'm', '--');
    set(h, 'FontSize', font_size);
    h = xlabel('Translation threshold in meter');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(class_id, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);    
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    hold on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TRANSLATIONS: X, Y, Z
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = errors_translation_x(index);
    fprintf('%20s, \tIndex:%d, \tMean X:%.2f [cm]\n', char(classes(class_id)), length(index), mean(D)*100)
    D = errors_translation_y(index);
    fprintf('%20s, \tIndex:%d, \tMean Y:%.2f [cm]\n', char(classes(class_id)), length(index), mean(D)*100)
    D = errors_translation_z(index);
    fprintf('%20s, \tIndex:%d, \tMean Z:%.2f [cm]\n', char(classes(class_id)), length(index), mean(D)*100)
    
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