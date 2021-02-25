using System;
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
        public ParametersStorage ParametersStorage { get; set; } = new ParametersStorage();
        public int FIn => ParametersStorage.Weights.Channels * KernelSize * KernelSize;
        public int FOut => ParametersStorage.Weights.Batch * KernelSize * KernelSize;
        
        //forward pass
        private Tensor _img2ColBuffer;
        private Tensor _dotBuffer;
        
        //dx
        private Tensor _paddingBuffer;
        private Tensor _img2ColDxBuffer;
        private Tensor _filters2DBuffer;
        private Tensor _rotBuffer;
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
            
            ParametersStorage.Weights = Builder.OfShape(wShape);
            ParametersStorage.Weights.Storage.Data = convLayerInfo.Weights;
            ParametersStorage.Gradients = Builder.OfShape(wShape.GetCopy());
            
            _initializer = new HeInitializer();
            
            InitializeBuffers();
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            ParametersStorage.Weights = Builder.OfShape(new Shape(FiltersCount, inputShape[1], KernelSize, KernelSize));
            _initializer.InitWeights(this);
            OutputShape = Tensor.GetConvolutionalShape(inputShape, ParametersStorage.Weights.Storage.Shape, Stride, 0);
            ParametersStorage.Gradients = Builder.OfShape(ParametersStorage.Weights.Storage.Shape);
            InitializeBuffers();
        }

        private void InitializeBuffers()
        {
            _img2ColBuffer = Builder.Empty();
            _dotBuffer = Builder.Empty();
                
            _paddingBuffer = Builder.Empty();
            _img2ColDxBuffer = Builder.Empty();
            _dxDotBuffer = Builder.Empty();
            _rotBuffer = Builder.Empty();
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
                WeightsShape = new ShapeInfo(ParametersStorage.Weights.Storage.Shape),
                Weights = ParametersStorage.Weights.Storage.Data
            };
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.Convolution(ParametersStorage.Weights, Stride, _img2ColBuffer, _dotBuffer, Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            OutputGradient = tensor;
            Input.ConvolutionDw(ParametersStorage.Weights, OutputGradient, _dy2DBuffer, _dwDotBuffer, _img2ColBuffer, ParametersStorage.Gradients);
            if (Prev != null)    
            {
                Input.ConvolutionDx(ParametersStorage.Weights, OutputGradient, _paddingBuffer, _img2ColDxBuffer, _filters2DBuffer, _rotBuffer, _dxDotBuffer, InputGradient);
                return InputGradient;
            }
            return null;
        }
        
    }
}
