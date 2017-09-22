function Do_Augmentation()
dogDir = 'data/dog';
endoscopeDir = 'data/endoscope';
mdir = endoscopeDir;
% dogDir = pwd;
Maxnum = 60000; %%%训练每类最多样本数
Minnum = 9000; %%%训练每类最少样本数，你自己调整下
% CropAug = 50;

[~, imageNames] = textread(fullfile(mdir, 'images.txt'), '%d %s');
imageNames = cellfun(@(x) strrep(x,'_',' '),imageNames,'UniformOutput',false);
[~, classLabel] = textread(fullfile(mdir, 'image_class_labels.txt'), '%d %d');
label = reshape(classLabel, 1, numel(classLabel));


[~, imageSet] = textread(fullfile(mdir, 'train_test_split.txt'), '%d %d');

index_Te = find(imageSet~=1);
index_Tr = find(imageSet==1);

index_ALL = [];
imageNames_ALL = {};

% index_Tr = [1:length(imageSet)]';
imageNames_Tr = imageNames(index_Tr);
classLabel_Tr = classLabel(index_Tr);
load('AugOpt', 'opt');
CLabel = unique(classLabel_Tr);

for i = 1:length(CLabel)
    disp(i);
    index = find(classLabel_Tr == i);
    if length(index) >= Maxnum
        NAug = ceil(length(index)/Maxnum);
        %找到当前类下的训练、评价、测试样本的索引
        index_train = find(ismember(imageSet(index_Tr(index)),1));
        index_val = find(ismember(imageSet(index_Tr(index)),2));
        index_test = find(ismember(imageSet(index_Tr(index)),3));
        %从训练、评价、测试样本中按5：3：2随机选取Maxnum个样本
        ind_train = randperm(length(index_train));
%         ind_train = ind_train(1:floor(Maxnum*0.5));
        ind_train = ind_train(1:floor(length(ind_train)/NAug));
        ind_train = index_train(ind_train);
        ind_val = randperm(length(index_val));
%         ind_val = ind_val(1:floor(Maxnum*0.3));
        ind_val = ind_val(1:floor(length(ind_val)/NAug));
        ind_val = index_val(ind_val);
        ind_test = randperm(length(index_test));
%         ind_test = ind_test(1:floor(Maxnum*0.2));
        ind_test = ind_test(1:floor(length(ind_test)/NAug));
        ind_test = index_test(ind_test);
        index_ALL = [index_ALL; index_Tr(index([ind_train; ind_val; ind_test]))];
        imageNames_ALL = [imageNames_ALL; imageNames_Tr(index([ind_train; ind_val; ind_test]))];
    else if length(index) <= Minnum
            NAug = ceil(Minnum/length(index));
            
            %设定切、加色、旋转、加躁的翻倍数
            CropAug = 0;ColorAug = 0;RotAug = 0;NoiseAug= 0;BlockAug= 0;MixAug = 100;
%             CropAug = 10;ColorAug = 1;RotAug = 5;
            imageNames_Tr_c = cellfun(@(x) fullfile(mdir, 'images', x), imageNames_Tr(index), 'UniformOutput', 0);
                
            N1 = min(NAug, CropAug);
            for k = 1:length(opt)
                opt(k).NAug =  N1; opt(k).transformation = 'f5' ;
            end
            [~, NewName] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0); 
            A = repmat(index_Tr(index(:)),1,N1)';
            index_ALL = [index_ALL;A(:)];

            N2 = min(NAug - N1,ColorAug);
            for k = 1:length(opt)
                opt(k).NAug =  N2; opt(k).transformation = 'Color' ;
            end
            [~, Tmp] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);
            NewName{1} =  [NewName{1}, Tmp{1}];
            A = repmat(index_Tr(index(:)),1,N2)';
            index_ALL = [index_ALL;A(:)];

            N3 = min(NAug - N1-N2,RotAug);
            for k = 1:length(opt)
                opt(k).NAug =  N3; opt(k).transformation = 'Rot' ;
            end
            [~, Tmp] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);
            NewName{1} =  [NewName{1}, Tmp{1}];
            A = repmat(index_Tr(index(:)),1,N3)';
            index_ALL = [index_ALL;A(:)];
            
            N4 = min(NAug - N1-N2-N3,NoiseAug);
            for k = 1:length(opt)
                opt(k).NAug =  N4; opt(k).transformation = 'Noise' ;
            end
            [~, Tmp] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);
            NewName{1} =  [NewName{1}, Tmp{1}];
            A = repmat(index_Tr(index(:)),1,N4)';
            index_ALL = [index_ALL;A(:)];
            
            N5 = min(NAug - N1-N2-N3-N4,BlockAug);
            for k = 1:length(opt)
                opt(k).NAug =  N5; opt(k).transformation = 'Block' ;
            end
            [~, Tmp] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);
            NewName{1} =  [NewName{1}, Tmp{1}];
            A = repmat(index_Tr(index(:)),1,N5)';
            index_ALL = [index_ALL;A(:)];
            
            N6 = min(NAug - N1-N2-N3-N4-N5,MixAug);
            for k = 1:length(opt)
                opt(k).NAug =  N6; opt(k).transformation = 'Mix' ;
            end
            [~, Tmp] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0);
            NewName{1} =  [NewName{1}, Tmp{1}];
            A = repmat(index_Tr(index(:)),1,N6)';
            index_ALL = [index_ALL;A(:)];
            
%             A = (repmat(index_Tr(index(:)), 1, NAug))';
%             index_ALL = [index_ALL;A(:)];
            imageNames_ALL = [imageNames_ALL; NewName{1}'];
% % %             
% % % % %             imo = imdb_get_batch_bcnn(images, varargin)
        else
            index_ALL = [index_ALL;index_Tr(index)];
            imageNames_ALL = [imageNames_ALL;imageNames_Tr(index)];
        end
    end
end


% imageNames_ALL = [imageNames_ALL; imageNames(index_Te)];
% index_ALL = [index_ALL; index_Te];

%% do it for endoscope dataset

% for the test dataset
imageNames_Te = imageNames(index_Te);
classLabel_Te = classLabel(index_Te);
load('AugOpt', 'opt');
CLabel = unique(classLabel_Te);

for i = 1:length(CLabel)
    disp(i);
    index = find(classLabel_Te == i);
    NAug = ceil(length(index_Te)/length(index)-1);

    imageNames_Te_c = cellfun(@(x) fullfile(mdir, 'images', x), imageNames_Te(index), 'UniformOutput', 0);

    N1 = NAug;
    for k = 1:length(opt)
        opt(k).NAug =  N1; opt(k).transformation = 'f25' ;
    end
    [~, NewName] = imdb_get_batch_bcnn1(imageNames_Te_c, opt, 'prefetch',0); 
    A = repmat(index_Te(index(:)),1,N1)';
    index_ALL = [index_ALL;A(:)];

    imageNames_ALL = [imageNames_ALL; NewName{1}'];
end


% for the validation dataset
t = numel(index_ALL)+1;

imageNames_Tr = imageNames(index_Tr);
classLabel_Tr = classLabel(index_Tr);
load('AugOpt', 'opt');
CLabel = unique(classLabel_Tr);

for i = 1:length(CLabel)
    disp(i);
    index = find(classLabel_Tr == i);
    NAug = ceil(length(index_Tr)/length(index)-1);

    imageNames_Tr_c = cellfun(@(x) fullfile(mdir, 'images', x), imageNames_Tr(index), 'UniformOutput', 0);

    N1 = NAug;
    for k = 1:length(opt)
        opt(k).NAug =  N1; opt(k).transformation = 'f25' ;
    end
    [~, NewName] = imdb_get_batch_bcnn1(imageNames_Tr_c, opt, 'prefetch',0); 
    A = repmat(index_Tr(index(:)),1,N1)';
    
    t_NewName = NewName{1}';
    t_A = A(:);
    ind = randperm(length(t_NewName)/5*3)';
%     index_ALL = [index_ALL;A(:)];
%     imageNames_ALL = [imageNames_ALL; NewName{1}'];
    index_ALL = [index_ALL;t_A(ind)];
    imageNames_ALL = [imageNames_ALL; t_NewName(ind)];
end

%%



imageNames_ALL = cellfun(@(x) strrep(x,'\','/'),imageNames_ALL,'UniformOutput',false);
imageNames_ALL = cellfun(@(x) strrep(x, [mdir, '/images/'],''),imageNames_ALL,'UniformOutput',false);

% % % % 
% % % % %%%%下面用原来　的格式　写入文件名和索引就好
% % % % textwrite('images_aug.txt', index_All)
% % % % textwrite('index_aug.txt', imageNames_ALL)


%%% Bounding boxes
fid_box = fopen(fullfile(mdir, 'bounding_boxes_new.txt'),'w');
temp = cellfun(@(x) fullfile(mdir,'images', x), imageNames_ALL, 'UniformOutput', 0);


%%% Image names
fid_image = fopen(fullfile(mdir, 'images_new.txt'),'w');
imageNames_ALL = cellfun(@(x) strrep(x,' ','_'),imageNames_ALL,'UniformOutput',false);
% % for i = 1:numel(imageNames_ALL)
% %     fprintf(fid_image,'%d %s\n',i,imageNames_ALL(i));
% % end

%%% Class labels
fid_labels = fopen(fullfile(mdir, 'image_class_labels_new.txt'),'w');
newlabel = label(index_ALL);
% % for i = 1:numel(imageNames_ALL)
% %     fprintf(fid_labels,'%d %d\n',i,newlabel(i));
% % end

%%% Image sets
fid_set = fopen(fullfile(mdir, 'train_test_split_new.txt'),'w');
newset = imageSet(index_ALL);
% do it for endoscope dataset
newset(t:end) = 2;


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
fid_ind = fopen(fullfile(mdir, 'index_all.txt'),'w');
fprintf(fid_ind,'%d\n',index_ALL);
fclose(fid_ind);