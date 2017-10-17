function Lap = GetLaplace(data_N, ts_index,ts_label)
% % if nargin < 3
% %     type = 'N';
% %     K = 20;
% % end
% % Lapstr = [type '-' num2str(K) '-s.mat'];
% % ImageN = [LapName '-ImgC-' Lapstr];
% % savePath = fullfile(opts.expDir,'Bilinear_CNN_similar.mat');


% for the dog dataset
savePath = fullfile('Wopt','SimMatrix','Bilinear_CNN_similar_dog.mat');
% for the endoscope dataset
% savePath = fullfile('Wopt','SimMatrix','Bilinear_CNN_similar_endoscope.mat');

try
    load(savePath, 'G');
catch
    if isempty(data_N)
        fea_path = @(ep) fullfile('Wopt','nonftbcnn', sprintf('bcnn_nonft_%06d.mat',ep));
%         fea = cell(numel(ts_index),1);
        fea = arrayfun(@(x) maxpool(fea_path(x)),ts_index,'UniformOutput',false);
        data_N = cell2mat(fea)';
        [~,i] = sort(ts_index);
        data_N = data_N(i,:);
        ts_label = ts_label(i);
    end
    
    G = sparse(GetSimMatrix(data_N, ts_label));
    save(savePath, 'G','-v7.3');
end

%%%refine by TM
% eps = 1e-3;
% diag(G) = ones(size(G, 1), 1) * eps;
% 
% 
% x = sparse(rand(4,4));
% k = sub2ind(size(G), [1:size(G,1)], [1:size(G, 1)]);
% G(k) = eps;
% n = size(G,1);
% G = G + speye(n,n)*eps;
% % % % index = find(G~=0);
% % % % G(index) = exp(-G(index));
% % % % %%%refine by TM



D = diag(sum(G, 2));
Lap = D-G;





function each_class_imageCNN_s = GetSimMatrix(ddata_N, label_index)

cluster_idx = label_index';
        
each_class_image_idx = arrayfun(@(x) find(cluster_idx == x),1:max(cluster_idx),'UniformOutput',false);
each_class_image_vector = cellfun(@(x) ddata_N(x,:),each_class_image_idx,'UniformOutput',false);
each_class_imageCNN_d = cellfun(@getSim,each_class_image_vector,'UniformOutput',false);

x_idx = cell(length(each_class_imageCNN_d),1);
y_idx = cell(length(each_class_imageCNN_d),1);
s = cell(length(each_class_imageCNN_d),1);
for i = 1:length(each_class_imageCNN_d)
    len = length(each_class_image_idx{i});
    x_idx{i,1} = repmat(each_class_image_idx{i},len,1);
    y_idx{i,1} = repmat(each_class_image_idx{i},1,len)';
    y_idx{i,1} = y_idx{i,1}(:);
    s{i,1} = each_class_imageCNN_d{i}(:);
end
x_idx = cell2mat(x_idx);
y_idx = cell2mat(y_idx);
s = double(cell2mat(s));
each_class_imageCNN_s = sparse(x_idx,y_idx,s,length(cluster_idx),length(cluster_idx));
clear x_idx;
clear y_idx;
clear s;




%         save(['ImgC/ImgC-' type '-' num2str(k) '-s.mat'],'each_class_imageCNN_s')
disp('... save similar matrix ...');




function x =getSim(x)
x = exp(-EuDist2(x,x));
eps = 1e-3;
x = reshape(mapminmax(x(:)', eps, 1),size(x));



function fea = maxpool(mfilename)
load(mfilename,'code');
code = reshape(code,[512,512]);
code = vl_nnpool(code,[8,8],'stride',[8,8],'method','max','pad',[0,1,0,1]);
[h,w] = size(code);
fea = reshape(code,h*w,1);

function res = EuDist2(X, B)
nbase = size(B, 1);
nframe = size(X, 1);
% find k nearest neighbors
XX = sum(X.*X, 2);
BB = sum(B.*B, 2);

res  = repmat(XX, 1, nbase)-2*X*B'+repmat(BB', nframe, 1);

