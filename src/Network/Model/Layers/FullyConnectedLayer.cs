using System;
using System.Collections.Generic;
using System.Text.Json;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class FullyConnectedLayer : BaseLayer, IParameterizedLayer
    {
        private readonly IWeightsInitializer _initializer;
        public int NeuronsCount { get; }
        public Tensor Weights { get; private set; }
        public Tensor Biases { get; private set; }
        public Tensor WeightsGradient { get; private set; }
        public int FIn => InputShape.Dimensions[3];
        public int FOut => NeuronsCount;

        private Tensor _iterationDw;
        
        public Dictionary<string, Tensor> Parameters { get; set; }
        
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
            Weights = Builder.OfShape(wShape);
            Weights.Storage.SetData(fullyLayerInfo.Weights);
            
            WeightsGradient = Builder.OfShape(wShape.GetCopy());
            _iterationDw = Builder.OfShape(wShape.GetCopy());
            
            _initializer = new HeInitializer();
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);

            Weights = Builder.OfShape(new Shape(1, 1, inputShape.Dimensions[3], NeuronsCount));
            _initializer.InitWeights(this);
            
            WeightsGradient = Builder.OfShape(new Shape(1, 1, inputShape.Dimensions[3], NeuronsCount));
            _iterationDw = Builder.OfShape(new Shape(1, 1, inputShape.Dimensions[3], NeuronsCount));
            
            OutputShape = new Shape(inputShape.Dimensions[0], 1, 1, NeuronsCount);
        }

        public override LayerInfo GetLayerInfo()
        {
            var layerInfo = base.GetLayerInfo();
            return new ParameterizedLayerInfo(layerInfo)
            {
                WeightsShape = new ShapeInfo(Weights.Storage.Shape),
                Weights = Weights.Storage.Array
            };
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.Dot2D(Weights, Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            OutputGradient = tensor;
            Input.FullyConnectedDw(OutputGradient, _iterationDw);
            WeightsGradient.Sum(_iterationDw);
            if (Prev != null)
            {
                Input.FullyConnectedDx(Weights, OutputGradient, InputGradient);
                return InputGradient;
            }

            return null;

        }

    }
}
