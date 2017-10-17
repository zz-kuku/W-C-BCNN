function SampleWT = gettrainW(SampleW, weightNorm, isboost,W_S_INNC)
if nargin < 4
    W_S_INNC = 1;
end
if W_S_INNC ~= 1
    minW = 1;
    maxW = W_S_INNC;
    SampleW = mapminmax(SampleW, minW, maxW);
end
            
SampleWT = SampleW;
if nargin < 3
    isboost = 0;
end
if weightNorm
    SampleWT = SampleW ./ sum(SampleW);
    SampleWT = SampleWT * length(SampleWT);
end