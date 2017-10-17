function SampleWTT =  Wopt(trainIndex, Loss, ts_Label,imdb,a,b)
%%trainIndex; index for training
%%Loss;       Loss for training
%%ts_Label;   Category label for training
load('tinfo', 'setting', 'L')
trainId = setting.ts_idx_conf;
load('ts_label', 'ts_label')
WeightAvg = 0;


if nargin > 0
    trainId = trainIndex;
end
if nargin > 1
    L = Loss;
end
if nargin > 2
    ts_label= ts_Label;
end
ts_idx = trainId;


setting.featparaW{1} = 0.1; %%%alpha
setting.featparaW{2} = 1; %%beta
setting.featparaW{3} = 0; %%%
setting.MLRweight = 1;

if nargin > 4
    setting.featparaW{1} = a;
    setting.featparaW{2} = b;
end

VoteTid = [1:length(trainId)];
% TFstr = 'E:\Tanmin\project\wsmlr_code/TrainInfo\image_click_Dog283_0';
% load([TFstr '_img_Fea_Clickcount.mat'])%%%Using Click Weight

% for the dog dataset
load('image_click_Dog283_0_img_Fea_Clickcount.mat'); 
Data_W = sum(img_Fea_Clickcount, 2);
Data_W = Data_W./sqrt(sum(Data_W.^2));
% for the endoscope dataset
% load('endoscope_score.mat');
% Data_W = score./sqrt(sum(score.^2));
Data_W = Data_W';
Data_W = Data_W(1, imdb.images.index(trainId));

if setting.featparaW{3} < 0
    Data_W = log(Data_W);setting.featparaW{3} = abs(setting.featparaW{3});
end
data_fea1 = '';
Lap = GetLaplace(data_fea1, trainId, ts_label);
Lap = (Lap + Lap') / 2;
clear 'data_fea1';
SampleWTT = ones(length(VoteTid), 1);
lb = zeros(size(SampleWTT));
% % Lap = Lap(ts_idx, ts_idx);
if ~isempty(Lap)
Laptmp = Lap(VoteTid, VoteTid);
else
Laptmp = [];
end
ClassW = ones(1, length(VoteTid));
SampleW = ones(1, length(VoteTid));
data_W = gettrainW(Data_W(VoteTid).*ClassW, setting.weightNorm, setting.isboost, setting.W_S_INNC);
ub = ones(size(SampleW))*length(VoteTid);                      
if setting.MLRweight
    [xx, yy, zz] = unique(ts_label(VoteTid));
    Aeq = zeros(length(xx), length(VoteTid));
     beq = [];
     for kkkt = 1:length(xx)
        index = (find(zz == kkkt)) ;
        data_W(index) = gettrainW(data_W(index).*ClassW(index), setting.weightNorm, setting.isboost, setting.W_S_INNC);
        Aeq(kkkt, index) = ones(size(index'));
       
        if ~WeightAvg
            ub(index) = (length(index));
            beq(kkkt) = length(index);
            data_W(index) =(length(index))* data_W(index) / sum(data_W(index) );
       
        end
      % ub(index) = length(VoteTid) / length(xx);
     end
     ClassW = getWclass(ts_label(VoteTid));
    SampleWT = gettrainW(SampleW.*ClassW, setting.weightNorm, setting.isboost, setting.W_S_INNC);
  
    if WeightAvg
        data_W = gettrainW(data_W.*ClassW, setting.weightNorm, setting.isboost, setting.W_S_INNC);
    beq = ones(length(xx), 1) * length(VoteTid) / length(xx);
        for kkkt = 1:length(xx)
        index = (find(zz == kkkt)) ; ub(index) = length(VoteTid) / length(xx);
        end
    end
    
else
    Aeq = ones(size(SampleW));
    beq = length(VoteTid);
    SampleWT = SampleW;
end
SampleWTT = SampleWT';data_W_T = data_W;                        




[H, f] = Getweightpara(setting.featparaW, setting.C/length(L)*L, Laptmp, data_W_T', SampleWTT);
clear 'Laptmp';
SampleWTT = quadprog(H,f,[],[],Aeq,beq,lb,ub);
                                        
                                        
function [H, f] = Getweightpara(featparaW, L, Lap, Click_0, SampleW)
if featparaW{1} < 0
    featparaW{1} = -featparaW{1};
    L = 0;
end

N = (length(SampleW));
H1 = sparse([1:N], [1:N], 1);
if ~isempty(Lap)
    H = 2*featparaW{2}*Lap+2*featparaW{1}*H1;
else
    H = 2*featparaW{1}*H1;
% eye(length(SampleW));
end
PW = (1-abs(featparaW{3}))*Click_0+abs(featparaW{3})*SampleW;
% if featparaW{3} < 0
    PW = PW / sum(PW) * length(PW);
% end
f = L - 2*featparaW{1}*(PW);