function y = vl_nnbilinear2maxpooling(x, param, varargin)


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

bs = size(x,4);

if backMode
    [h2,w2,~,bs] = size(dzdy);
    y = reshape(dzdy,[1,1,w2*h2,bs]);
else
    y = reshape(x,[512,512,1,bs]);
end
