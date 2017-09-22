classdef Bilinear2MaxPooling < dagnn.ElementWise
  properties
    param = 1e-8;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnbilinear2maxpooling(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnbilinear2maxpooling(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function obj = Bilinear2MaxPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
