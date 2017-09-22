function fea = Maxpooling(code)

kernel = [4,4];
stride = [4,4];

[f, n] = size(code);

f_sq = fix(sqrt(f));
if sqrt(f) ~= f_sq
    disp('error!')
    return;
end



temp_code = reshape(code,f_sq,f_sq,n);

try
    code = vl_nnpool(temp_code, kernel, 'stride', stride, 'method', 'max', 'pad', [0,1,0,1]);
catch
    setup;
    code = vl_nnpool(temp_code, kernel, 'stride', stride, 'method', 'max', 'pad', [0,1,0,1]);
end
[h,w,n] = size(code);
fea = reshape(code,h*w,n);