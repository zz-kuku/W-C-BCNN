function y = vl_nnmaxpooling2bilinear(x, param, varargin)


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

[h,w,~,bs] = size(x);

if backMode
    y = reshape(dzdy,[h,w,1,bs]);
else
    y = reshape(x,[1,1,h*w,bs]);
end
