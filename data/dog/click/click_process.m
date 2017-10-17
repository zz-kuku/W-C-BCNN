function click_process()

load('image_click_Dog283_0_img_Fea_Clickcount.mat');
load('image_click_Dog283_0_click_nonN-C-k-10-20-c11_ND_S_data.mat','data_label');

each_image_clickcount = sum(img_Fea_Clickcount,2);

each_image_clickcount = log(each_image_clickcount);

% each_image_clickcount = each_image_clickcount./sum(each_image_clickcount).*length(each_image_clickcount);


if exist('./imdb-seed.mat','file')
    imdb = load('imdb-seed.mat');
    index = imdb.images.index;
    each_image_clickcount_t = each_image_clickcount(unique(index));
    data_label = data_label(unique(index));
elseif exist('../index_all.txt','file');
    index = textread('../index_all.txt','%d');
    each_image_clickcount_t = each_image_clickcount(unique(index));
    data_label = data_label(unique(index));
end


e = 0.5;
a = unique(data_label);
for u = 1:numel(a)
    i = a(u);
    idx = data_label==i;
    each_image_clickcount_t(idx) = mapminmax(each_image_clickcount_t(idx)',e,2-e);
end

each_image_clickcount(unique(index)) = each_image_clickcount_t;

save('each_image_clickcount','each_image_clickcount');

