using System;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class Softmax : BaseLayer
    {
        private Tensor _maxBuffer;
        
        public Softmax(){}

        public Softmax(LayerInfo info) : base(info)
        {
            _maxBuffer = Builder.Empty();
        }

        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            _maxBuffer = Builder.Empty();
            OutputShape = inputShape.GetCopy();
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.Softmax(Output, _maxBuffer);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            if (Prev == null)
                return null;
            
            OutputGradient = tensor;
            Output.SoftmaxDx(OutputGradient, InputGradient);
            return InputGradient;
        }
    }
}
