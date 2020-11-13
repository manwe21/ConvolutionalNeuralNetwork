using System;
using Network.NeuralMath;

namespace Network.Serialization
{
    [Serializable]
    public class LayerInfo
    {
        public string LayerType { get; set; }
        public ShapeInfo InputShape { get; set; }
        public ShapeInfo OutputShape { get; set; }

        public LayerInfo(){}
        
        public LayerInfo(LayerInfo info)
        {
            LayerType = info.LayerType;
            InputShape = info.InputShape;
            OutputShape = info.OutputShape;
        }
        
    }

    [Serializable]
    public class ParameterizedLayerInfo : LayerInfo
    {
        public ShapeInfo WeightsShape { get; set; }
        public float[] Weights { get; set; }

        public ParameterizedLayerInfo(LayerInfo info) : base(info) { }
        
    }    
    
    [Serializable]
    public class ActivationLayerInfo : LayerInfo
    {
        public string FunctionType { get; set; }
        
        public ActivationLayerInfo(LayerInfo info) : base(info) { }
        
    }

    [Serializable]
    public class ConvolutionLayerInfo : ParameterizedLayerInfo
    {
        public int FiltersCount { get; set; }
        public int Stride { get; set; }
        public int KernelSize { get; set; }
        
        public ConvolutionLayerInfo(LayerInfo info) : base(info) { }
    }

    [Serializable]
    public class PoolingLayerInfo : LayerInfo
    {
        public int PoolingSize { get; set; }
        public int Stride { get; set; }
        
        public PoolingLayerInfo(LayerInfo info) : base(info) { }
    }

    [Serializable]
    public class PaddingLayerInfo : LayerInfo
    {
        public int Padding { get; set; }
        
        public PaddingLayerInfo(LayerInfo info) : base(info) { }
    }
    
    
}    
