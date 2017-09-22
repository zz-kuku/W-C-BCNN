

alpha = [0.01, 0.1, 1, 10];
beta = [0.001, 0.01, 0.1, 1, 10];

flag = 1; % 1.origin ,2.the fix large   ,3.mapminmax


% seed = '08';

result = cell(3,1);


% % % [a,b,c] = fileparts(opts.resultPath) ;
% % % txtPath = fullfile(a, [b '.txt']) ;
% % % f=fopen(txtPath, 'w') ;
% % % fprintf(f, '%s', str) ;
% % % fclose(f) ;

% temp_seed = {{'08','09','14'},{'10','11'},{'12','13'}};
temp_seed = {{},{},{'03'}};

for flag = 1:3
for k = 1:numel(temp_seed{flag});
    seed = temp_seed{flag}{k};
for i = 1:numel(alpha)
    a = alpha(i);
    for j = 1:numel(beta)
        b = beta(j);
        
        savedir = ['data/exp/dog-seed-' seed '/Maxpooling_' num2str(a) '_' num2str(b) '_noWA_' num2str(flag)];
        
        if exist(savedir,'dir')
            mpath = fullfile(savedir, 'result-bcnnvdmft.mat');
            info = load(mpath) ;
            str = {} ;
% %             str{end+1} = sprintf('data: %s', ['data/exp/dog-seed-' seed] ) ;
% %             str{end+1} = sprintf(' setup: %10s', opts.suffix) ;
            str{end+1} = sprintf(['alpha: ' num2str(a) ', beta: ' num2str(b) ', flag: ' num2str(flag)]);
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
            
            result{flag}(i,j) = info.test.acc*100;
        end
        
        
        
        
        
% % %         mkdir(savedir);
% % %         data_W = Wopt(trainIndex, Loss, ts_Label,imdb,a,b);
% % %         switch flag
% % %             case 2
% % %                 t = 2;
% % %                 data_W(data_W>t) = t;
% % %             case 3
% % %                 data_W = mapminmax(data_W',0,2)';
% % %         end
% % %         
% % %         save(['data/exp/dog-seed-' seed '/data_W.mat'], 'data_W');
% % %         run_experiments;
        
        
% % %         fn = dir(['data/exp/dog-seed-' seed '/svmtrain*']);
% % %         for k = 1:length(fn)
% % %             dos(['mv ' fullfile(['data/exp/dog-seed-' seed],fn(k).name) ' ' fullfile(savedir, fn(k).name)]);
% % % %             dos(['rm ' fullfile(['data/exp/dog-seed-' seed],fn(k).name)]);
% % %         end
% % %         fn = dir(['data/exp/dog-seed-' seed '/result*']);
% % %         for k = 1:length(fn)
% % %             dos((['mv ' fullfile(['data/exp/dog-seed-' seed],fn(k).name) ' ' fullfile(savedir, fn(k).name)]));
% % % %             dos(['rm ' fullfile(['data/exp/dog-seed-' seed],fn(k).name)]);
% % %         end
% % %         dos(['mv ' fullfile(['data/exp/dog-seed-' seed],'data_W.mat') ' ' fullfile(savedir, 'data_W.mat')]);
% % % %         dos(['rm ' fullfile(['data/exp/dog-seed-' seed],'data_W.mat')]);
        
    end
end
end
end
save('result.mat','result');


for flag = 1:3
    fid = fopen(['result' num2str(flag) '.txt'],'w');
    for i = 1:numel(alpha)
        for j = 1:numel(beta)
            fprintf(fid,'\t%.2d',result{flag}(i,j));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
end
