classdef MaxPooling2Bilinear < dagnn.ElementWise
  properties
    param = 1e-8;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnmaxpooling2bilinear(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnmaxpooling2bilinear(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function obj = MaxPooling2Bilinear(varargin)
      obj.load(varargin) ;
    end
  end
end
