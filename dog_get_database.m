function imdb = dog_get_database(dogDir, useCropped, useVal)
% Automatically change directories
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).
if useCropped
    imdb.imageDir = fullfile(dogDir, 'images_cropped') ;
else
    imdb.imageDir = fullfile(dogDir, 'images');
end

imdb.clickDir = fullfile(dogDir,'click');

imdb.maskDir = fullfile(dogDir, 'masks'); % doesn't exist
imdb.sets = {'train', 'val', 'test'};

% Class names
[~, classNames] = textread(fullfile(dogDir, 'classes.txt'), '%d %s');
classNames = cellfun(@(x) strrep(x,'_',' '),classNames,'UniformOutput',false);
imdb.classes.name = horzcat(classNames(:));

% Image names
[~, imageNames] = textread(fullfile(dogDir, 'images_new.txt'), '%d %s');
imageNames = cellfun(@(x) strrep(x,'_',' '),imageNames,'UniformOutput',false);
imdb.images.name = imageNames;
imdb.images.id = (1:numel(imdb.images.name));

% Class labels
[~, classLabel] = textread(fullfile(dogDir, 'image_class_labels_new.txt'), '%d %d');
imdb.images.label = reshape(classLabel, 1, numel(classLabel));

% Bounding boxes
[~,x, y, w, h] = textread(fullfile(dogDir, 'bounding_boxes_new.txt'), '%d %f %f %f %f');
imdb.images.bounds = round([x y x+w-1 y+h-1]');

% INDEX
if exist(fullfile(dogDir,'index_all.txt'),'file');
    index = textread(fullfile(dogDir,'index_all.txt'),'%d');
    imdb.images.index = index;
end

% Image sets
[~, imageSet] = textread(fullfile(dogDir, 'train_test_split_new.txt'), '%d %d');
imdb.images.set = zeros(1,length(imdb.images.id));
imdb.images.set(imageSet == 1) = 1;
imdb.images.set(imageSet == 2) = 2;
imdb.images.set(imageSet == 3) = 3;

if useVal
    rng(0)
    trainSize = numel(find(imageSet==1));
    
    trainIdx = find(imageSet==1);
    
    % set 1/3 of train set to validation
    valIdx = trainIdx(randperm(trainSize, round(trainSize/3)));
    imdb.images.set(valIdx) = 2;
end

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.images.difficult = false(1, numel(imdb.images.id)) ; 
