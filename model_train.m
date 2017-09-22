function model_train(varargin)

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

[opts, imdb] = model_setup1(varargin{:}) ;
opts.isClickW =1;

% -------------------------------------------------------------------------
%                                          Train encoders and compute codes
% -------------------------------------------------------------------------

str = '';
AllSplit = 0;
if isempty(opts.FRound{2})
    AllSplit = 1;
end
if opts.FRound{1}~=1
    range = cell(1, length(opts.FRound{2}));
    if isempty(opts.FRound{2})	
        opts.FRound{2} = [1:opts.FRound{1}];
    end
    for j = opts.FRound{2}
        str = ['S-', num2str(opts.FRound{1}), '-', num2str(j)];
        for i = 1:numel(opts.encoders)
            opts.encoders{i}.codePath_s{j} = opts.encoders{i}.codePath;
%             opts.encoders{i}.path_s{j} = opts.encoders{i}.path;
            [a,b,c] = fileparts(opts.encoders{i}.codePath_s{j});
            opts.encoders{i}.codePath_s{j} = fullfile(a, [b, str, c]);
%             [a,b,c] = fileparts(opts.encoders{i}.path_s{j});
%             opts.encoders{i}.path_s{j} = fullfile(a, [b, str, c]);
        end
        nround = ceil(length(imdb.images.id) / opts.FRound{1});
        range{j} = [(j -1)*nround+1:min(j*nround, ...
            length(imdb.images.id))];
          
    end
else
    opts.FRound{2}=1;
    range{1} = [1:length(imdb.images.id)];
        for i = 1:numel(opts.encoders)
            opts.encoders{i}.codePath_s{1} = opts.encoders{i}.codePath;
%             opts.encoders{i}.path_s{1} = opts.encoders{i}.path;
        end
end

      
switch opts.dataAugmentation
    case 'none'
        ts = 1;
    case 'f2'
        ts = 2;
    otherwise
        error('not supported data augmentation')
end 
if ~exist(opts.resultPath)
  psi = {} ;
  for i = 1:numel(opts.encoders)
for j = opts.FRound{2}
    if exist(opts.encoders{i}.codePath_s{j})
      load(opts.encoders{i}.codePath_s{j}, 'code', 'area') ;
    else
        
      if exist(opts.encoders{i}.path)
        encoder = load(opts.encoders{i}.path) ;
        if isa(encoder.net, 'dagnn.DagNN'), encoder.net = dagnn.DagNN.loadobj(encoder.net); end
        if isfield(encoder, 'net')
            if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
            encoder.net = net_move_to_device(encoder.net, device);
        end
      else
        opts.encoders{i}.opts = horzcat(opts.encoders{i}.opts);
        train = find(ismember(imdb.images.set, [1 2])) ;
        train = vl_colsubset(train, 1000, 'uniform') ;
        encoder = encoder_train_from_images(...
          imdb, imdb.images.id(train), ...
          opts.encoders{i}.opts{:}, ...
          'useGpu', opts.useGpu, ...
          'scale', opts.imgScale) ;
%       encoder.net.useGpu = opts.useGpu;
        encoder_save(encoder, opts.encoders{i}.path) ;
      end
      code = encoder_extract_for_images(encoder, imdb, imdb.images.id(range{j}), 'dataAugmentation', opts.dataAugmentation, 'scale', opts.imgScale) ;
      savefast(opts.encoders{i}.codePath_s{j}, 'code') ;
    end
    if AllSplit
% % % if (opts.clickconn)      
        code1 = code(1:end-4318, :);
        code2 = code(end-4317:end, :);
        code = Maxpooling(code1);
        code = cat(1, code, code2);
        clear code1 code2;
% % % end        
        psi{i}(:, [range{j}(end)*ts-length(range{j})*ts+1:range{j}(end)*ts]) = code ;
    end
    clear code ;
end
  end
  psi = cat(1, psi{:}) ;
end

%if ~isempty(opts.FRound{2})
if ~AllSplit
    return;
end
% -------------------------------------------------------------------------
%                                                            Train and test
% -------------------------------------------------------------------------

if exist(opts.resultPath)
  info = load(opts.resultPath) ;
else
  info = traintest(opts, imdb, psi) ;
  if ~isempty(opts.FRound{3})
      return;
  end
  save(opts.resultPath, '-struct', 'info') ;
  vl_printsize(1) ;
  [a,b,c] = fileparts(opts.resultPath) ;
  print('-dpdf', fullfile(a, [b '.pdf'])) ;
end

str = {} ;
str{end+1} = sprintf('data: %s', opts.expDir) ;
str{end+1} = sprintf(' setup: %10s', opts.suffix) ;
str{end+1} = sprintf(' mAP: %.1f', info.test.map*100) ;
if isfield(info.test, 'acc')
  str{end+1} = sprintf(' acc: %6.1f ', info.test.acc*100);
end
if isfield(info.test, 'im_acc')
  str{end+1} = sprintf(' acc wo normlization: %6.1f ', info.test.im_acc*100);
end
str{end+1} = sprintf('\n') ;
str = cat(2, str{:}) ;
fprintf('%s', str) ;

[a,b,c] = fileparts(opts.resultPath) ;
txtPath = fullfile(a, [b '.txt']) ;
f=fopen(txtPath, 'w') ;
fprintf(f, '%s', str) ;
fclose(f) ;



% -------------------------------------------------------------------------
function info = traintest(opts, imdb, psi)
% -------------------------------------------------------------------------

% Train using verification or not
verificationTask = isfield(imdb, 'pairs');

dataW = ones(numel(imdb.images.set),1);
% click weight
if opts.isClickW
    load(fullfile(opts.expDir,'data_W'));
    dataW(ismember(imdb.images.set, 1)) = data_W ;
end

switch opts.dataAugmentation
    case 'none'
        ts =1 ;
    case 'f2'
        ts = 2;
    otherwise
        error('not supported data augmentation')
end

if verificationTask, 
    train = ismember(imdb.pairs.set, [1 2]) ;
    test = ismember(imdb.pairs.set, 3) ;
else % classification task
    multiLabel = (size(imdb.images.label,1) > 1) ; % e.g. PASCAL VOC cls
    train = ismember(imdb.images.set, [1 2]) ;
    train = repmat(train, ts, []);
    train = train(:)';
    test = ismember(imdb.images.set, 3) ;
    test = repmat(test, ts, []);
    test = test(:)';
    info.classes = find(imdb.meta.inUse) ;
    
    % Train classifiers
    C = 1 ;
    w = {} ;
    b = {} ;
    
    
    c_max = numel(info.classes);
    c_s =1;c_e=c_max;
    
    %为了多个MATLAB并行跑多类
%     if ~isempty(opts.FRound{3}) || opts.FRound{1} == 1
%         c_s = (opts.FRound{3}-1)*ceil(c_max/opts.FRound{1})+1;
%         c_e = min(c_max ,opts.FRound{3}*ceil(c_max/opts.FRound{1}));
    
%     if ~exist(fullfile(opts.expDir,['svmtrain_283.mat']))
        
    parfor c= c_s:c_e
      fprintf('\n-------------------------------------- ');
      fprintf('OVA-classifier: class: %d\n', c) ;
      if ~multiLabel
        y = 2*(imdb.images.label == info.classes(c)) - 1 ;
      else
        y = imdb.images.label(c,:) ;
      end
      y_test = y(test(1:ts:end));
      y = repmat(y, ts, []);
      y = y(:)';
      np = sum(y(train) > 0) ;
      nn = sum(y(train) < 0) ;
      n = np + nn ;
      

      [w{c},b{c}] = vl_svmtrain(psi(:,train & y ~= 0), y(train & y ~= 0), 1/(n* C), ...
        'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
        'maxNumIterations', n * 200, 'Weights', dataW(train & y ~= 0 ,1) ) ;

      pred = w{c}'*psi + b{c} ;
      

      % try cheap calibration
      mp = median(pred(train & y > 0)) ;
      mn = median(pred(train & y < 0)) ;
      b{c} = (b{c} - mn) / (mp - mn) ;
      w{c} = w{c} / (mp - mn) ;
      pred = w{c}'*psi + b{c} ;

      scores{c} = pred ;
     
      pred_test = reshape(pred(test), ts, []);
      pred_test = mean(pred_test, 1);

      [~,~,i]= vl_pr(y(train), pred(train)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y_test, pred_test) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(train), pred(train), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
      [~,~,i]= vl_pr(y_test, pred_test, 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
      
      
      
%       if ~exist(fullfile(opts.expDir,['svmtrain_' num2str(c) '.mat']),'file') && (~mod(c,ceil(c_max/opts.FRound{1})) || ~mod(c,c_max))
%           save(fullfile(opts.expDir,['svmtrain_' num2str(c) '.mat']),'scores','w','b','ap','ap11','tap','tap11','nap','tnap');
%       end
      
    end
    
%     end
    
    %为了多个MATLAB并行跑多类
%         if opts.FRound{1}~=1
%             return;
%         end
%     end
    
%     if opts.FRound{1}~=1
%         for i = 1:ceil(c_max/opts.FRound{1}):c_max
%             c = min(i+ceil(c_max/opts.FRound{1}-1),c_max);
%             load(fullfile(opts.expDir,['svmtrain_' num2str(c) '.mat']),'scores','w','b','ap','ap11','tap','tap11','nap','tnap');
%             scores_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = scores(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             w_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = w(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             b_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = b(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             ap_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = ap(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             ap11_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = ap11(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             tap_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = tap(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             tap11_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = tap11(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             nap_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = nap(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%             tnap_t(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max)) = tnap(i:min(i+ceil(c_max/opts.FRound{1}-1),c_max));
%         end
%         scores = scores_t;
%         w = w_t;
%         b = b_t;
%         ap = ap_t;
%         tap = tap_t;
%         ap11 = ap11_t;
%         tap11 = tap11_t;
%         nap = nap_t;
%         tnap = tnap_t;
%     end
    
    % Book keeping
    info.w = cat(2,w{:}) ;
    info.b = cat(2,b{:}) ;
    info.scores = cat(1, scores{:}) ;
    info.train.ap = ap ;
    info.train.ap11 = ap11 ;
    info.train.nap = nap ;
    info.train.map = mean(ap) ;
    info.train.map11 = mean(ap11) ;
    info.train.mnap = mean(nap) ;
    info.test.ap = tap ;
    info.test.ap11 = tap11 ;
    info.test.nap = tnap ;
    info.test.map = mean(tap) ;
    info.test.map11 = mean(tap11) ;
    info.test.mnap = mean(tnap) ;
    clear ap nap tap tnap scores ;
    fprintf('mAP train: %.1f, test: %.1f\n', ...
      mean(info.train.ap)*100, ...
      mean(info.test.ap)*100);

    % Compute predictions, confusion and accuracy
    [~,preds] = max(info.scores,[],1) ;
    info.testScores = reshape(info.scores(:,test), size(info.scores,1), ts, []);
    info.testScores = reshape(mean(info.testScores, 2), size(info.testScores,1), []);
    [~,pred_test] = max(info.testScores, [], 1);
    [~,gts] = ismember(imdb.images.label, info.classes) ;
    gts_test = gts(test(1:ts:end));
    gts = repmat(gts, ts, []);
    gts = gts(:)';

    [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(train), preds(train)) ;
    [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts_test, pred_test) ;
    
    
    [~, info.train.im_acc] = compute_confusion(numel(info.classes), gts(train), preds(train), ones(size(gts(train))), true) ;
    [~, info.test.im_acc] = compute_confusion(numel(info.classes), gts_test, pred_test, ones(size(gts_test)), true) ;
%     [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;
end

% -------------------------------------------------------------------------
function code = encoder_extract_for_images(encoder, imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.batchSize = 64 ;
opts.maxNumLocalDescriptorsReturned = 500 ;
opts.concatenateCode = true;
opts.dataAugmentation = 'none';
opts.scale = 1;
opts = vl_argparse(opts, varargin) ;

[~,imageSel] = ismember(imageIds, imdb.images.id) ;
imageIds = unique(imdb.images.id(imageSel)) ;
n = numel(imageIds) ;

% prepare batches
n = ceil(numel(imageIds)/opts.batchSize) ;
batches = mat2cell(1:numel(imageIds), 1, [opts.batchSize * ones(1, n-1), numel(imageIds) - opts.batchSize*(n-1)]) ;
batchResults = cell(1, numel(batches)) ;

switch opts.dataAugmentation
    case 'none'
        ts = 1;
    case 'f2'
        ts = 2;
    otherwise
        error('not supported data augmentation')
end
code = cell(1, numel(imageIds)*ts) ;

% just use as many workers as are already available
numWorkers = matlabpool('size') ;
%parfor (b = 1:numel(batches), numWorkers)
for b = numel(batches):-1:1
  batchResults{b} = get_batch_results(imdb, imageIds, batches{b}, ...
                        encoder, opts.maxNumLocalDescriptorsReturned, opts.dataAugmentation, opts.scale) ;
                    
  m = numel(batches{b});
  for j = 1:m
      k = batches{b}(j) ;
      for aa=1:ts
        code{(k-1)*ts+aa} = batchResults{b}.code{(j-1)*ts+aa};
      end
  end
  %%zz_uncle  ------ if the memory is not enough
  batchResults{b} = [];
  %%
end


if opts.concatenateCode
   code = cat(2, code{:}) ;
end
% code is either:
% - a cell array, each cell containing an array of local features for a
%   segment
% - an array of FV descriptors, one per segment

% -------------------------------------------------------------------------
function result = get_batch_results(imdb, imageIds, batch, encoder, maxn, dataAugmentation, scale)
% -------------------------------------------------------------------------
m = numel(batch) ;
im = cell(1, m) ;
task = getCurrentTask() ;
if ~isempty(task), tid = task.ID ; else tid = 1 ; end

switch dataAugmentation
    case 'none'
        tfs = [0 ; 0 ; 0 ];
    case 'f2'
        tfs = [...
            0   0 ;
            0   0 ;
            0   1];
    otherwise
        error('not supported data augmentation')
end

ts = size(tfs,2);
im = cell(1, m*ts);
for i = 1:m
    fprintf('Task: %03d: encoder: extract features: image %d of %d\n', tid, batch(i), numel(imageIds)) ;
    for j=1:ts
        idx = (i-1)*ts+j;
        im{idx} = imread(fullfile(imdb.imageDir, imdb.images.name{imdb.images.id == imageIds(batch(i))}));
        if size(im{idx}, 3) == 1, im{idx} = repmat(im{idx}, [1 1 3]); end; %grayscale image
        
        tf = tfs(:,j) ;
        if tf(3), sx = fliplr(1:size(im{idx}, 2)) ;
            im{idx} = im{idx}(:,sx,:);
        end
    end
end

if ~isfield(encoder, 'numSpatialSubdivisions')
  encoder.numSpatialSubdivisions = 1 ;
end
switch encoder.type
    case 'rcnn'
        net = vl_simplenn_tidy(encoder.net);
        net.useGpu = encoder.net.useGpu;
        code_ = get_rcnn_features(encoder.net, ...
            im, ...
            'regionBorder', encoder.regionBorder) ;
    case 'dcnn'
        gmm = [] ;
        if isfield(encoder, 'covariances'), gmm = encoder ; end
        code_ = get_dcnn_features(encoder.net, ...
            im, ...
            'encoder', gmm, ...
            'numSpatialSubdivisions', encoder.numSpatialSubdivisions, ...
            'maxNumLocalDescriptorsReturned', maxn, 'scales', scale) ;
    case 'dsift'
        gmm = [] ;
        if isfield(encoder, 'covariances'), gmm = encoder ; end
        code_ = get_dcnn_features([], im, ...
            'useSIFT', true, ...
            'encoder', gmm, ...
            'numSpatialSubdivisions', encoder.numSpatialSubdivisions, ...
            'maxNumLocalDescriptorsReturned', maxn) ;
    case 'bcnn'
       %% zz_uncle
        if exist(fullfile(imdb.clickDir,'image_click_Dog283_0_click_nonN-C-k-10-20-c11_ND_S_data.mat'),'file')
            load(fullfile(imdb.clickDir,'image_click_Dog283_0_click_nonN-C-k-10-20-c11_ND_S_data.mat'),'data_fea');
            data_fea = data_fea./repmat(sqrt(sum(data_fea.^2,2)),1,size(data_fea,2));
% %         if exist(fullfile(imdb.clickDir,'endoscope_click_feature_normalization.mat'),'file')
% %             load(fullfile(imdb.clickDir,'endoscope_click_feature_normalization.mat'),'data_fea');
            batch1 = imdb.images.index(batch);
            fea_click{1} = single(full(data_fea(batch1,: )'));
%             load(fullfile(imdb.clickDir,'each_image_clickcount.mat'),'each_image_clickcount');
%            fea_click{2} = single(full(each_image_clickcount(batch1,:)));
            fea_click{2} = single(ones(numel(batch1),1));
        else
            fea_click = [];
        end
        
        code_ = get_bcnn_features(encoder.net, im, fea_click, 'scales', scale);
end
result.code = code_ ;

% -------------------------------------------------------------------------
function encoder = encoder_train_from_images(imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.type = 'rcnn' ;
opts.model = '' ;
opts.modela = '';
opts.modelb = '';
opts.layer = 0 ;
opts.layera = 0 ;
opts.layerb = 0 ;
opts.useGpu = false ;
opts.regionBorder = 0.05 ;
opts.numPcaDimensions = +inf ;
opts.numSamplesPerWord = 1000 ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.renormalize = false ;
opts.numWords = 64 ;
opts.numSpatialSubdivisions = 1 ;
opts.normalization = 'sqrt_L2';
opts.scale = 1;
opts = vl_argparse(opts, varargin) ;

encoder.type = opts.type ;
encoder.regionBorder = opts.regionBorder ;
switch opts.type
  case {'dcnn', 'dsift'}
    encoder.numWords = opts.numWords ;
    encoder.renormalize = opts.renormalize ;
    encoder.numSpatialSubdivisions = opts.numSpatialSubdivisions ;
end

switch opts.type
    case {'rcnn', 'dcnn'}
        encoder.net = load(opts.model) ;
        if ~isempty(opts.layer)
            encoder.net.layers = encoder.net.layers(1:opts.layer) ;
        end
        if opts.useGpu
            encoder.net = vl_simplenn_tidy(encoder.net);
            encoder.net = vl_simplenn_move(encoder.net, 'gpu') ;
            encoder.net.useGpu = true ;
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            encoder.net = vl_simplenn_move(encoder.net, 'cpu') ;
            encoder.net.useGpu = false ;
        end
   case 'bcnn'
       encoder.normalization = opts.normalization;
        encoder.neta = load(opts.modela);
        if isfield(encoder.neta, 'net')
            encoder.neta = encoder.neta.net;
        end
        
        if ~isempty(opts.modelb)
            assert(~isempty(opts.layerb), 'layerb is not specified')
            encoder.netb = load(opts.modelb);
            if isfield(encoder.netb, 'net')
                encoder.netb = encoder.netb.net;
            end
            encoder.netb.layers = encoder.netb.layers(1:opts.layerb);
        end
        
        if ~isempty(opts.layera)
            encoder.layera = opts.layera;
            maxLayer = opts.layera;
            if ~isempty(opts.layerb) && isempty(opts.modelb)
                maxLayer = max(maxLayer, opts.layerb);
                encoder.layerb = opts.layerb;
            end
            encoder.neta.layers = encoder.neta.layers(1:maxLayer);
        end
        
        if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
        
        encoder.neta = net_move_to_device(encoder.neta, device);
        if isfield(encoder, 'netb')
            encoder.netb = net_move_to_device(encoder.netb, device);
        end
        
        encoder.net = initializeNetFromEncoder(encoder);
        rmFields = {'neta', 'netb', 'layera', 'layerb'};
        rmIdx = find(ismember(rmFields, fieldnames(encoder)));
        for i=1:numel(rmIdx)
            encoder = rmfield(encoder, rmFields{rmIdx(i)});
        end
        if isa(encoder.net, 'dagnn.DagNN')
            encoder.net.mode = 'test';
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            if opts.useGpu
                encoder.net.useGpu = true;
	    else
		encoder.net.useGpu = false;
            end
        end        
end

switch opts.type
  case {'rcnn', 'bcnn'}
    return ;
end

% Step 0: sample descriptors
fprintf('%s: getting local descriptors to train GMM\n', mfilename) ;
code = encoder_extract_for_images(encoder, imdb, imageIds, 'concatenateCode', false, 'scale', opts.scale) ;
descrs = cell(1, numel(code)) ;
numImages = numel(code);
numDescrsPerImage = floor(encoder.numWords * opts.numSamplesPerWord / numImages);
for i=1:numel(code)
  descrs{i} = vl_colsubset(code{i}, numDescrsPerImage) ;
end
descrs = cat(2, descrs{:}) ;
fprintf('%s: obtained %d local descriptors to train GMM\n', ...
  mfilename, size(descrs,2)) ;


% Step 1 (optional): learn PCA projection
if opts.numPcaDimensions < inf || opts.whitening
  fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
  encoder.projectionCenter = mean(descrs,2) ;
  x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
  X = x*x' / size(x,2) ;
  [V,D] = eig(X) ;
  d = diag(D) ;
  [d,perm] = sort(d,'descend') ;
  d = d + opts.whiteningRegul * max(d) ;
  m = min(opts.numPcaDimensions, size(descrs,1)) ;
  V = V(:,perm) ;
  if opts.whitening
    encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
  else
    encoder.projection = V(:,1:m)' ;
  end
  clear X V D d ;
else
  encoder.projection = 1 ;
  encoder.projectionCenter = 0 ;
end
descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
if encoder.renormalize
  descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
end

% Step 2: train GMM
v = var(descrs')' ;
[encoder.means, encoder.covariances, encoder.priors] = ...
  vl_gmm(descrs, opts.numWords, 'verbose', ...
  'Initialization', 'kmeans', ...
  'CovarianceBound', double(max(v)*0.0001), ...
  'NumRepetitions', 1) ;
