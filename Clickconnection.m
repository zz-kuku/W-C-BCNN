% L2Norm is the dagnn wrapper which normalizes the features to be norm 1 in
% 2-norm at each location

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN. It uses MATCONVNET package and is made 
% available under the terms of the BSD license (see the COPYING file).

classdef Clickconnection < dagnn.ElementWise
  properties
    param = 1e-10;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnClickcon(inputs{1}, inputs{2}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derInputs{2}]= vl_nnClickcon(inputs{1}, inputs{2}, derOutputs{1}) ;
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Clickconnection(varargin)
      obj.load(varargin) ;
    end
  end
end


