function updateWeigh(epoch,imdb,trainIndex,loss)

[trainIndex ,i] = sort(trainIndex);
loss = loss(i);


trainLabel = imdb.images.label(trainIndex);

addpath(genpath('./Wopt'));

% train_ori_ind = imdb.images.index(trainIndex);

%%% update similar matrix
% % % netf = net;
% % % netf.layers = netf.layers(1:netf.getLayerIndex('drop_1'));
% % % netf.vars = netf.vars(1:netf.getVarIndex('d_1'));
% % % netf.params = netf.params(1:netf.getParamIndex('netb_conv5b'));
% % % 
% % % inputs = getBatchDagNNWrapper();
% % % netf.eval(input);



modelFigPath = fullfile(imdb.clickDir, ['dataW' num2str(epoch) '.pdf']) ;

dataW = Wopt(trainIndex,loss,trainLabel,imdb);

figure(2);
plot(dataW,'b-.*');

drawnow ;
print(2, modelFigPath, '-dpdf') ;

% load(fullfile(imdb.clickDir,'each_image_clickcount.mat'));
% each_image_clickcount(train_ori_ind) = dataW;
% % data_W = ones(numel(imdb.images.index),1);
% % data_W(trainIndex) = dataW;


data_W = dataW;
save(fullfile(imdb.clickDir,['data_W' num2str(epoch) '.mat']),'data_W');
% % save(fullfile(imdb.clickDir,'dataW.mat'),'dataW');

end


