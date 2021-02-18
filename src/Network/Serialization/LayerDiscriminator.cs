using System;

namespace Network.Serialization
{
    public enum LayerDiscriminator
    {
        LayerInfo,
        ParameterizedLayerInfo,
        ActivationLayerInfo,
        ConvolutionLayerInfo,
        PoolingLayerInfo,
        PaddingLayerInfo
    }

    public static class LayerDiscriminatorExtensions
    {
        public static Type GetLayerInfoType(this LayerDiscriminator discriminator)
        {
            switch (discriminator)
            {
                case LayerDiscriminator.ActivationLayerInfo:
                    return typeof(ActivationLayerInfo);
                case LayerDiscriminator.ConvolutionLayerInfo:
                    return typeof(ConvolutionLayerInfo);
                case LayerDiscriminator.PaddingLayerInfo:
                    return typeof(PaddingLayerInfo);
                case LayerDiscriminator.ParameterizedLayerInfo:
                    return typeof(ParameterizedLayerInfo);
                case LayerDiscriminator.PoolingLayerInfo:
                    return typeof(PoolingLayerInfo);
                default: return typeof(LayerInfo);
            }
        }
    }
    
}