% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = imdb_get_batch_bcnn(images, opts, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;
numAugments = size(im{1},4)/numel(batch);

labels = reshape(repmat(labels, numAugments, 1), 1, size(im{1},4));

if opts(1).isClickconn
    % the click feature of dog dataset
    load(fullfile(imdb.clickDir,'image_click_Dog283_0_click_nonN-C-k-10-20-c11_ND_S_data.mat'),'data_fea');
    data_fea = data_fea./repmat(sqrt(sum(data_fea.^2,2)),1,size(data_fea,2));
    % the click feature of endoscope dataset
%     load(fullfile(imdb.clickDir,'endoscope_click_feature_normalization.mat'),'data_fea');
    batch1 = imdb.images.index(batch);
    im{3}{1} = single(full(data_fea(batch1,: )'));
%    load(fullfile(imdb.clickDir,'each_image_clickcount.mat'),'each_image_clickcount');
%    im{3}{2} = single(full(each_image_clickcount(batch1,:)));
    im{3}{2} = single(ones(numel(batch1),1));
    switch opts(1).transformation
        case 'f2'
            im{3}{1} = repmat(im{3}{1},1,2);
            im{3}{2} = repmat(im{3}{2},2,1);
    end
    if useGpu
        im3{1} = gpuArray(im{3}{1});
        im3{2} = gpuArray(im{3}{2});
    else
        im3{1} = im{3}{1};
        im3{2} = im{3}{2};
    end
end


if nargout > 0
  if useGpu
    im1 = gpuArray(im{1}) ;
    im2 = gpuArray(im{2}) ;
  else
      im1 = im{1};
      im2 = im{2};
  end
  if ~opts(1).isClickconn
    inputs = {'input', im1, 'netb_input', im2, 'label', labels} ;
  else
    inputs = {'input', im1, 'netb_input', im2, 'label', labels, 'click_vector',im3} ;
  end
end

