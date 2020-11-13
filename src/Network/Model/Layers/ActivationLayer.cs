using System;
using System.Text.Json;
using Network.NeuralMath;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class ActivationLayer : BaseLayer
    {
        public IFunction ActivationFunction { get; }
        
        public ActivationLayer(IFunction activationFunction)
        {
            ActivationFunction = activationFunction;
        }

        public ActivationLayer(LayerInfo info) : base(info)
        {
            var actLayerInfo = info as ActivationLayerInfo;
            if(actLayerInfo == null)
                throw new ArgumentException(nameof(info));
            
            //TODO Find better solution to work with functions with parameters
            var functionType = Type.GetType(actLayerInfo.FunctionType);
            if (functionType == null)
                throw new ArgumentException(nameof(info));
            
            ActivationFunction = (IFunction) Activator.CreateInstance(functionType);
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            OutputShape = inputShape.GetCopy();
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.Activation(ActivationFunction, Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            if (Prev == null)
                return null;
            
            OutputGradient = tensor;
            Input.ActivationDx(ActivationFunction, tensor, InputGradient);
            return InputGradient;
        }

        public override LayerInfo GetLayerInfo()
        {
            var layerInfo = base.GetLayerInfo();
            return new ActivationLayerInfo(layerInfo)
            {
                FunctionType = ActivationFunction.GetType().FullName
            };
        }
    }
}
