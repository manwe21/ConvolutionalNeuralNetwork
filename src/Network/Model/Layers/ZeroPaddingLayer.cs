using System;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class ZeroPaddingLayer : BaseLayer
    {
        public int Padding { get; }
        
        public ZeroPaddingLayer(int padding)
        {
            Padding = padding;
        }

        public ZeroPaddingLayer(LayerInfo info) : base(info)
        {
            var padLayerInfo = info as PaddingLayerInfo;
            if (padLayerInfo == null)
                throw new ArgumentException(nameof(info));
            
            Padding = padLayerInfo.Padding;
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            OutputShape = Tensor.GetPaddingShape(inputShape, Padding);
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.Pad(Padding, Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            if (Prev == null)
                return null;
            
            OutputGradient = tensor;
            Input.PadDx(Padding, OutputGradient, InputGradient);
            return InputGradient;
        }

        public override LayerInfo GetLayerInfo()
        {
            var layerInfo = base.GetLayerInfo();
            return new PaddingLayerInfo(layerInfo)
            {
                Padding = this.Padding
            };
        }
    }
}
