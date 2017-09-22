% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
labels = imdb.images.label(batch) ;
opts(1).isAugmentation = ismember(labels, find(opts(1).isAugmentation));
opts(2).isAugmentation = ismember(labels, find(opts(1).isAugmentation));

im = imdb_get_batch_bcnn(images, opts, ...
                            'prefetch', nargout == 0);
                        
t = numel(find(opts(1).isAugmentation));                       
if t>0
    numAugments = (size(im{1},4)-numel(batch)+t)/t;
    labels = [reshape(labels(~opts(1).isAugmentation),1,numel(labels(~opts(1).isAugmentation))) ,reshape(repmat(labels(opts(1).isAugmentation), numAugments, 1), 1, size(im{1},4)-numel(batch)+t)];
else                    
    numAugments = size(im{1},4)/numel(batch);
    labels = reshape(repmat(labels, numAugments, 1), 1, size(im{1},4));
end



if nargout > 0
  if useGpu
    im1 = gpuArray(im{1}) ;
    im2 = gpuArray(im{2}) ;
  else
      im1 = im{1};
      im2 = im{2};
  end
  inputs = {'input', im1, 'netb_input', im2, 'label', labels} ;
end

