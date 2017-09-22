

% alpha = [0.01,0.1];
% % alpha = [1,10];
% % beta = [0.001,0.1,1,10];
% beta = [0.001];

alpha = [0.01,0.1,1,10];
beta = [0.001,0.01,0.1,1,10];


flag = 2; % 1.origin ,2.the fix large   ,3.mapminmax

addpath('Wopt');
load('Wopt/test_endoscope.mat');

seed = '03';

Split = {3,[],[]};

for i = 1:numel(alpha)
    a = alpha(i);
    for j = 1:numel(beta)
        b = beta(j);
        
        savedir = ['data/exp/endoscope-seed-' seed '/Maxpooling_' num2str(a) '_' num2str(b) '_noWA'];
        mkdir(savedir);
        data_W = Wopt(trainIndex, Loss, ts_Label,imdb,a,b);
        switch flag
            case 2
                t = 2;
                loc = data_W>t;
                data_W(loc) = mean(data_W(loc));
            case 3
                data_W = mapminmax(data_W',0,2)';
        end
        
        save(['data/exp/endoscope-seed-' seed '/data_W.mat'], 'data_W');
        run_experiments;
        
        
        fn = dir(['data/exp/endoscope-seed-' seed '/svmtrain*']);
        for k = 1:length(fn)
            dos(['mv ' fullfile(['data/exp/endoscope-seed-' seed],fn(k).name) ' ' fullfile(savedir, fn(k).name)]);
%             dos(['rm ' fullfile(['data/exp/endoscope-seed-' seed],fn(k).name)]);
        end
        fn = dir(['data/exp/endoscope-seed-' seed '/result*']);
        for k = 1:length(fn)
            dos((['mv ' fullfile(['data/exp/endoscope-seed-' seed],fn(k).name) ' ' fullfile(savedir, fn(k).name)]));
%             dos(['rm ' fullfile(['data/exp/endoscope-seed-' seed],fn(k).name)]);
        end
        dos(['mv ' fullfile(['data/exp/endoscope-seed-' seed],'data_W.mat') ' ' fullfile(savedir, 'data_W.mat')]);
%         dos(['rm ' fullfile(['data/exp/endoscope-seed-' seed],'data_W.mat')]);
        
    end
end



