function Do_Augmentation()
dogDir = 'data/dog';
% dogDir = pwd;
Maxnum = 3000; %%%训练每类最多样本数
Minnum = 500;%%%训练每类最少样本数，你自己调整下

[~, imageNames] = textread(fullfile(dogDir, 'images.txt'), '%d %s');
imageNames = cellfun(@(x) strrep(x,'_',' '),imageNames,'UniformOutput',false);
[~, classLabel] = textread(fullfile(dogDir, 'image_class_labels.txt'), '%d %d');
label = reshape(classLabel, 1, numel(classLabel));


[~, imageSet] = textread(fullfile(dogDir, 'train_test_split.txt'), '%d %d');

index_Te = find(imageSet~=1);
imageNames_ALL = imageNames(index_Te);
index_ALL = index_Te;

index_Tr = find(imageSet==1);
imageNames_Tr = imageNames(index_Tr);
classLabel_Tr = classLabel(index_Tr);
load('AugOpt', 'opt');
CLabel = unique(classLabel_Tr);
for i = 1:length(CLabel)
    disp(i);
    index = find(classLabel_Tr == i);
    if length(index) > Maxnum
        ind = randperm(length(index));
        ind = ind(1:Maxnum);
        index_ALL = [index_ALL; index_Tr(index(ind))];
        imageNames_ALL = [imageNames_ALL; imageNames_Tr(index(ind))];
    else if length(index) < Minnum
            NAug = ceil(Minnum/length(index));
%             NAug = 70;
            imageNames_Tr_c = cellfun(@(x) fullfile('data/dog/images', x), imageNames_Tr(index), 'UniformOutput', 0);
            if NAug <= 50
                %%%前５０个跑f25，后面跑rotation+noise
                for k = 1:length(opt)
                    opt(k).NAug =  NAug; opt(k).transformation = 'f25' ;
                end
                [~, NewName] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);  
            else
                for k = 1:length(opt)
                    opt(k).NAug =  50; opt(k).transformation = 'f25' ;
                end
                [~, NewName] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);         
                for k = 1:length(opt)
                    opt(k).NAug =  NAug - 50; opt(k).transformation = 'Rot' ;
                end
                [~, Tmp] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);
                 NewName{1} =  [NewName{1}, Tmp{1}];
            end
            A = (repmat(index_Tr(index(:)), 1, NAug))';
            index_ALL = [index_ALL;A(:)];
            imageNames_ALL = [imageNames_ALL; NewName{1}'];
            
% %             imo = imdb_get_batch_bcnn(images, varargin)
        else
            index_ALL = [index_ALL;index_Tr(index)];
            imageNames_ALL = [imageNames_ALL;imageNames_Tr(index)];
        end
    end
end

imageNames_ALL = cellfun(@(x) strrep(x,'\','/'),imageNames_ALL,'UniformOutput',false);
imageNames_ALL = cellfun(@(x) strrep(x,'data/dog/images/',''),imageNames_ALL,'UniformOutput',false);

% % % % 
% % % % %%%%下面用原来　的格式　写入文件名和索引就好
% % % % textwrite('images_aug.txt', index_All)
% % % % textwrite('index_aug.txt', imageNames_ALL)


%%% Bounding boxes
fid_box = fopen('data/dog/bounding_boxes_new.txt','w');
temp = cellfun(@(x) fullfile('data/dog/images', x), imageNames_ALL, 'UniformOutput', 0);


%%% Image names
fid_image = fopen('data/dog/images_new.txt','w');
imageNames_ALL = cellfun(@(x) strrep(x,' ','_'),imageNames_ALL,'UniformOutput',false);
% % for i = 1:numel(imageNames_ALL)
% %     fprintf(fid_image,'%d %s\n',i,imageNames_ALL(i));
% % end

%%% Class labels
fid_labels = fopen('data/dog/image_class_labels_new.txt','w');
newlabel = label(index_ALL);
% % for i = 1:numel(imageNames_ALL)
% %     fprintf(fid_labels,'%d %d\n',i,newlabel(i));
% % end

%%% Image sets
fid_set = fopen('data/dog/train_test_split_new.txt','w');
newset = imageSet(index_ALL);

% % % for i = 1:numel(imageNames_ALL)
% % %     fprintf(fid_set,'%d %d\n',i,newset(i));
% % % end




for i = 1:numel(imageNames_ALL)
%     temp = fullfile('data/dog/images',imageNames_ALL(i))
    [b,a] = size(imread(temp{i})) ;
    fprintf(fid_box,'%d 1.0 1.0 %.1f %.1f\n',i,a,b);
    fprintf(fid_set,'%d %d\n',i,newset(i));
    fprintf(fid_image,'%d %s\n',i,imageNames_ALL{i});
    fprintf(fid_labels,'%d %d\n',i,newlabel(i));
    disp(i);
end
fclose(fid_set);
fclose(fid_box);
fclose(fid_labels);
fclose(fid_image);


%%%index
fid_ind = fopen('data/dog/index_all.txt','w');
fprintf(fid_ind,'%d\n',index_ALL);
fclose(fid_ind);