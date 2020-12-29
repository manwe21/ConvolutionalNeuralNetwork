using System;
using System.Collections.Generic;
using System.Text.Json;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class ConvolutionLayer : BaseLayer, IParameterizedLayer
    {
        private readonly IWeightsInitializer _initializer;
        
        public int FiltersCount { get; }
        public int KernelSize { get; }
        public int Stride { get; }
        
        public Tensor Weights { get; private set; }
        public Tensor Biases { get; private set; }
        public Tensor WeightsGradient { get; set; }
        
        public int FIn => Weights.Channels * KernelSize * KernelSize;
        public int FOut => Weights.Batch * KernelSize * KernelSize;
        
        public Dictionary<string, Tensor> Parameters { get; set; }

        //Memory buffers for intermediate computations
        
        private Tensor _iterationDw;
        
        //forward pass
        private Tensor _img2ColBuffer;
        
        //dx
        private Tensor _paddingBuffer;
        private Tensor _img2ColDxBuffer;
        private Tensor _filters2DBuffer;
        private Tensor _dxDotBuffer;
        
        //dw
        private Tensor _dy2DBuffer;
        private Tensor _dwDotBuffer;
        
        public ConvolutionLayer(int filtersCount, int kernelSize, int stride)
        {
            KernelSize = kernelSize;
            Stride = stride;
            FiltersCount = filtersCount;
            _initializer = new HeInitializer();
        }
        
        public ConvolutionLayer(int filtersCount, int kernelSize, int stride, IWeightsInitializer initializer)
        {
            KernelSize = kernelSize;
            Stride = stride;
            FiltersCount = filtersCount;
            _initializer = initializer;
        }

        public ConvolutionLayer(LayerInfo info) : base(info)
        {
            var convLayerInfo = info as ConvolutionLayerInfo;
            if(convLayerInfo == null)
                throw new ArgumentException(nameof(info));
            
            var wShape = new Shape(convLayerInfo.WeightsShape.B, convLayerInfo.WeightsShape.C, convLayerInfo.WeightsShape.H, convLayerInfo.WeightsShape.W);

            FiltersCount = convLayerInfo.FiltersCount;
            Stride = convLayerInfo.Stride;
            KernelSize = convLayerInfo.KernelSize;
            
            Weights = Builder.OfShape(wShape);
            Weights.Storage.SetData(convLayerInfo.Weights);
            WeightsGradient = Builder.OfShape(wShape.GetCopy());
            _iterationDw = Builder.OfShape(wShape.GetCopy());
            
            _initializer = new HeInitializer();
            
            InitializeBuffers();
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            Weights = Builder.OfShape(new Shape(FiltersCount, inputShape[1], KernelSize, KernelSize));
            _initializer.InitWeights(this);
            OutputShape = Tensor.GetConvolutionalShape(inputShape, Weights.Storage.Shape, Stride, 0);
            WeightsGradient = Builder.OfShape(Weights.Storage.Shape);
            _iterationDw = Builder.OfShape(Weights.Storage.Shape);
            InitializeBuffers();
        }

        private void InitializeBuffers()
        {
            _img2ColBuffer = Builder.Empty();
                
            _paddingBuffer = Builder.Empty();
            _img2ColDxBuffer = Builder.Empty();
            _dxDotBuffer = Builder.Empty();
            _filters2DBuffer = Builder.Empty();
                
            _dy2DBuffer = Builder.Empty();
            _dwDotBuffer = Builder.Empty();
        }

        public override LayerInfo GetLayerInfo()
        {
            var layerInfo = base.GetLayerInfo();
            return new ConvolutionLayerInfo(layerInfo)
            {
                FiltersCount = this.FiltersCount,
                Stride = this.Stride,
                KernelSize = this.KernelSize,
                WeightsShape = new ShapeInfo(Weights.Storage.Shape),
                Weights = Weights.Storage.Array
            };
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.Convolution(Weights, Stride, 0, _img2ColBuffer, Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            OutputGradient = tensor;
            Input.ConvolutionDw(Weights, OutputGradient, _dy2DBuffer, _dwDotBuffer, _img2ColBuffer, _iterationDw);
            WeightsGradient.Sum(_iterationDw);
            
            if (Prev != null)    
            {
                Input.ConvolutionDx(Weights, OutputGradient, _paddingBuffer, _img2ColDxBuffer, _filters2DBuffer, _dxDotBuffer, InputGradient);
                return InputGradient;
            }
            return null;
        }

        public static ConvolutionLayer LoadLayer(Dictionary<string, float> layerData)
        {
            return new ConvolutionLayer(1, 1, 1);
        }
        
    }
}
