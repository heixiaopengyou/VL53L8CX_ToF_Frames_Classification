function Augmented_Data2C = Augmentations (data_table)
if nargin < 1
        data_table = load('dataTable_4Class.mat').dataTable;
end
rotation90 = {};
rotation180 = {};
rotation270 = {};
left_flip = {};
top_flip = {};
top_crop = {};
bottom_crop = {};
left_crop = {};
right_crop = {};
translate_right = {};
translate_left = {};
center_crop = {};
sequences_train = {};
tof = data_table.ToFReading;
zonal = data_table.ZonalData;
classes = data_table.Gesture;
for i = 1:numel(tof)
    frame_data = tof{i};
    sample_data = zonal{i};
    sample_data = mapminmax(sample_data);
    frame_data(frame_data>400) = 0;
    frame_data = mapminmax(frame_data);
    left_flip_data = fliplr(frame_data);
    top_flip_data = flipud(frame_data);
    left_flip_zonal = fliplr(sample_data);
    top_flip_zonal = flipud(sample_data);
    % Rotated frames
    % rotation45_data = imrotate(roi_data, 45);
    rotation90_data = imrotate(frame_data, 90);
    rotation180_data = imrotate(frame_data, 180);
    rotation270_data = imrotate(frame_data, 270);
    rotation90_zonal = imrotate(sample_data, 90);
    rotation180_zonal = imrotate(sample_data, 180);
    rotation270_zonal = imrotate(sample_data, 270);
    % Cropped frames
    % Cropped frames
    [height, width] = size(frame_data);
    top_crop_data = frame_data(1:height-1, :);
    top_crop_zonal = sample_data(1:height-1, :);
    % size(top_crop_data)
    bottom_crop_data = frame_data(2:end, :);
    bottom_crop_zonal = sample_data(2:end, :);
    % size(bottom_crop_data)
    left_crop_data = frame_data(:, 1:width-1);
    left_crop_zonal = sample_data(:, 1:width-1);
    % size(left_crop_data)
    right_crop_data = frame_data(:, 2:end);
    right_crop_zonal = sample_data(:, 2:end);
    % size(right_crop_data)
    translate_right_data = imtranslate(frame_data,[1, 1],OutputView="full");
    translate_right_zonal = imtranslate(sample_data,[1, 1],OutputView="full");
    % size(translate_right_data)
    translate_left_data = imtranslate(frame_data,[-1, -1],OutputView="full");
    translate_left_zonal = imtranslate(sample_data,[-1, -1],OutputView="full");
    % size(translate_left_data)
    center_crop_data = frame_data(2:height - 2, 2:width - 2);
    center_crop_zonal = sample_data(2:height - 2, 2:width - 2);
    % size(center_crop_data)
    frame_data = cast(frame_data, 'single');
    sample_data = cast(sample_data, 'single');
    top_crop_data = imresize(top_crop_data, [8, 8], 'bicubic');
    top_crop_data = cast(top_crop_data, 'single');
    top_crop_zonal = imresize(top_crop_zonal, [8, 8], 'bicubic');
    top_crop_zonal = cast(top_crop_zonal, 'single');
    bottom_crop_data = imresize(bottom_crop_data, [8, 8], 'bicubic');
    bottom_crop_data = cast(bottom_crop_data, 'single');
    bottom_crop_zonal = imresize(bottom_crop_zonal, [8, 8], 'bicubic');
    bottom_crop_zonal = cast(bottom_crop_zonal, 'single');
    left_crop_data = imresize(left_crop_data, [8, 8], 'bicubic');
    left_crop_data = cast(left_crop_data, 'single');
    left_crop_zonal = imresize(left_crop_zonal, [8, 8], 'bicubic');
    left_crop_zonal = cast(left_crop_zonal, 'single');
    right_crop_data = imresize(right_crop_data, [8, 8], 'bicubic');
    right_crop_data = cast(right_crop_data, 'single');
    right_crop_zonal = imresize(right_crop_zonal, [8, 8], 'bicubic');
    right_crop_zonal = cast(right_crop_zonal, 'single');
    center_crop_data = imresize(center_crop_data, [8, 8], 'bicubic');
    center_crop_data = cast(center_crop_data, 'single');
    center_crop_zonal = imresize(center_crop_zonal, [8, 8], 'bicubic');
    center_crop_zonal = cast(center_crop_zonal, 'single');
    left_flip_data = cast(left_flip_data, 'single');
    top_flip_data = cast(top_flip_data, 'single');
    left_flip_zonal = cast(left_flip_zonal, 'single');
    top_flip_zonal = cast(top_flip_zonal, 'single');
    translate_right_data = imresize(translate_right_data, [8, 8], 'bicubic');
    translate_right_data = cast(translate_right_data, 'single');
    translate_left_data = imresize(translate_left_data, [8, 8], 'bicubic');
    translate_left_data = cast(translate_left_data, 'single');
    translate_right_zonal = imresize(translate_right_zonal, [8, 8], 'bicubic');
    translate_right_zonal = cast(translate_right_zonal, 'single');
    translate_left_zonal = imresize(translate_left_zonal, [8, 8], 'bicubic');
    translate_left_zonal = cast(translate_left_zonal, 'single');
    rotation90_data = cast(rotation90_data, 'single');
    rotation180_data = cast(rotation180_data, 'single');
    rotation270_data = cast(rotation270_data, 'single');
    rotation90_zonal = cast(rotation90_zonal, 'single');
    rotation180_zonal = cast(rotation180_zonal, 'single');
    rotation270_zonal = cast(rotation270_zonal, 'single');
    top_flip{end+1} = cat(3,top_flip_data, top_flip_zonal);
    rotation90{end+1} = cat(3,rotation90_data, rotation90_zonal);
    rotation180{end+1} = cat(3,rotation180_data, rotation180_zonal);
    rotation270{end+1} = cat(3,rotation270_data, rotation270_zonal);
    left_flip{end+1} = cat(3,left_flip_data, left_flip_zonal);
    top_crop{end+1} = cat(3,top_crop_data, top_crop_zonal);
    bottom_crop{end+1} = cat(3,bottom_crop_data, bottom_crop_zonal);
    left_crop{end+1} = cat(3,left_crop_data, left_crop_zonal);
    right_crop{end+1} = cat(3,right_crop_data, right_crop_zonal);
    translate_right{end+1} = cat(3,translate_right_data, translate_right_zonal);
    translate_left{end+1} = cat(3,translate_left_data, translate_left_zonal);
    center_crop{end+1} = cat(3,center_crop_data, center_crop_zonal);
    sequences_train{end+1} = cat(3,frame_data, sample_data);
end
dataset = [rotation90 rotation180 rotation270 left_flip top_flip top_crop bottom_crop left_crop right_crop translate_right translate_left center_crop sequences_train];
labels_train = classes;
Labels = [labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train; labels_train];
Labels = transpose(Labels);
Augmented_Data2C = table(dataset', Labels', 'VariableNames',["Augmented_data", "Labels"]);
end
