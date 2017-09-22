function [y, y_click] = vl_nnClickcon(x1, x2, varargin)



backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;

if backMode
  dzdy = varargin{1} ;
end

if issparse(x2{1})
    x2{1} = single(full(x2{1}));
end

if issparse(x2{2})
    x2{2} = single(full(x2{2}));
end

[c_num, bs] = size(x2{1});

if backMode
    y1 = dzdy(1,1,1:size(x1,3),1:bs);
    Tmp = reshape(repmat(x2{2}', size(y1, 3), 1), [1, 1, size(y1,3), size(x2{2},1)]);
    y = y1.*Tmp;
    y_click = [];
else
    c = reshape(x2{1},1,1,c_num,bs);
    y = cat(3, x1, c);
%     y = y.*x2{2};
    
    Tmp = reshape(repmat(x2{2}', size(y, 3), 1), [1, 1, size(y,3), size(x2{2},1)]);
    y = Tmp.*y;
end
