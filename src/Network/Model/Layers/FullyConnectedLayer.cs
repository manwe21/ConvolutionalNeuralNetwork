using System;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class FullyConnectedLayer : BaseLayer, IParameterizedLayer
    {
        private readonly IWeightsInitializer _initializer;
        public int NeuronsCount { get; }
        public ParametersStorage ParametersStorage { get; set; } = new ParametersStorage();
        public int FIn => InputShape[3];
        public int FOut => NeuronsCount;

        private Tensor _transBufferDx;
        private Tensor _transBufferDw;
        
        public FullyConnectedLayer(int neuronsCount, IWeightsInitializer initializer)
        {
            NeuronsCount = neuronsCount;
            _initializer = initializer;
        }
        
        public FullyConnectedLayer(int neuronsCount)
        {
            NeuronsCount = neuronsCount;
            _initializer = new HeInitializer();
        }

        public FullyConnectedLayer(LayerInfo info) : base(info)
        {
            var fullyLayerInfo = info as ParameterizedLayerInfo;
            if(fullyLayerInfo == null)
                throw new ArgumentException(nameof(info));
            
            var wShape = new Shape(fullyLayerInfo.WeightsShape.B, fullyLayerInfo.WeightsShape.C, fullyLayerInfo.WeightsShape.H, fullyLayerInfo.WeightsShape.W);
            ParametersStorage.Weights = Builder.OfShape(wShape);
            ParametersStorage.Weights.Storage.Data = fullyLayerInfo.Weights;
            
            ParametersStorage.Gradients = Builder.OfShape(wShape.GetCopy());
            
            _initializer = new HeInitializer();
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            ParametersStorage.Weights = Builder.OfShape(new Shape(1, 1, inputShape[3], NeuronsCount));
            _initializer.InitWeights(this);
            ParametersStorage.Gradients = Builder.OfShape(new Shape(1, 1, inputShape[3], NeuronsCount));
            OutputShape = new Shape(inputShape[0], 1, 1, NeuronsCount);

            _transBufferDx = Builder.Empty();
            _transBufferDw = Builder.Empty();
        }

        public override LayerInfo GetLayerInfo()
        {
            var layerInfo = base.GetLayerInfo();
            return new ParameterizedLayerInfo(layerInfo)
            {
                WeightsShape = new ShapeInfo(ParametersStorage.Weights.Storage.Shape),
                Weights = ParametersStorage.Weights.Storage.Data
            };
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            var w = ParametersStorage.Weights;
            Input.Dot2D(w, Input.Batch, Input.Width, w.Height, w.Width, OutputShape, Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            OutputGradient = tensor;
            Input.FullyConnectedDw(OutputGradient, _transBufferDw, ParametersStorage.Gradients);
            if (Prev != null)
            {
                Input.FullyConnectedDx(ParametersStorage.Weights, OutputGradient, _transBufferDx, InputGradient);
                return InputGradient;
            }

            return null;

        }

    }
}
