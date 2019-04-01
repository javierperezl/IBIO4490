function [bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

SVM_thres = -0.2;
template_size = feature_params.template_size;
hog_cell_size = feature_params.hog_cell_size;
win_size = template_size / hog_cell_size;

for i = 1:length(test_scenes)
    
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0);
    cur_image_ids = {};
      
    
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
 
    scales = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]; %Se modifica para obtener dif escalas
    
    for sc = scales
        img_sc = imresize(img, sc);
        
        hog = vl_hog(img_sc, hog_cell_size); %HOG A IMG REESCALADA
        HOG_rows = size(hog,1);
        HOG_cols = size(hog,2);
        for ii = 1:HOG_rows - win_size + 1
            for jj = 1:HOG_cols - win_size + 1
                
                crop_win = hog(ii:ii + win_size - 1, jj:jj + win_size - 1, :);
                crop_win = reshape(crop_win,1, win_size^2 * 31);
                confidence = crop_win * w + b; %USANDO MODELO SVM ENTRENADO, w y b 
                
                if confidence > SVM_thres
                    xy_min_c = ((jj - 1) * hog_cell_size + 1) / sc;
                    xy_min_r = ((ii - 1) * hog_cell_size + 1) / sc;
                    xy_max_c = ((jj + win_size - 1) * hog_cell_size) / sc;
                    xy_max_r = ((ii + win_size - 1) * hog_cell_size) / sc;
                    
                    cur_bboxes = [cur_bboxes; [xy_min_c, xy_min_r, xy_max_c, xy_max_r]];
                    cur_confidences = [cur_confidences; confidence];
                end
            end
        end
    end
    
    num_patches = size(cur_bboxes, 1);
    cur_image_ids(1:num_patches,1) = {test_scenes(i).name};
    
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
    
    %Addapted from https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj5/html/jkim844/index.html
end