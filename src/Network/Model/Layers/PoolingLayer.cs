using System;
using System.Text.Json;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class PoolingLayer : BaseLayer
    {
        public int PoolingSize { get; }
        public int Stride { get; }
        
        private Tensor _maxIndexes;
        
        public PoolingLayer(int poolingSize, int stride)
        {
            PoolingSize = poolingSize;
            Stride = stride;
        }

        public PoolingLayer(LayerInfo info) : base(info)
        {
            var poolLayerInfo = info as PoolingLayerInfo;
            if(poolLayerInfo == null)
                throw new ArgumentException(nameof(info));

            PoolingSize = poolLayerInfo.PoolingSize;
            Stride = poolLayerInfo.Stride;
            _maxIndexes = Builder.Empty();
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            _maxIndexes = Builder.Empty();
            OutputShape = Tensor.GetPoolingShape(inputShape, PoolingSize, Stride);
        }

        public override LayerInfo GetLayerInfo()
        {
            var layerInfo = base.GetLayerInfo();
            return new PoolingLayerInfo(layerInfo)
            {
                PoolingSize = this.PoolingSize,
                Stride = this.Stride
            };
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.MaxPool(PoolingSize, Stride, Output, _maxIndexes);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            if (Prev == null)
                return null;
            
            OutputGradient = tensor;
            Input.MaxPoolDx(OutputGradient, _maxIndexes, InputGradient);
            return InputGradient;
        }
    }
}
