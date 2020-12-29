using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{
    public class FlattenLayer : BaseLayer
    {
        public FlattenLayer() { }

        public FlattenLayer(LayerInfo info) : base(info) { }
        
        public override void Initialize(Shape inputShape)
        {
            base.Initialize(inputShape);
            OutputShape = Tensor.GetFlattenShape(inputShape);
        }

        public override Tensor Forward(Tensor tensor)
        {
            Input = tensor;
            Input.ToFlatten(Output);
            return Output;
        }

        public override Tensor Backward(Tensor tensor)
        {
            if (Prev == null)
                return null;
            
            OutputGradient = tensor;
            Input.FlattenDx(OutputGradient, InputGradient);
            return InputGradient;
        }
    }
}
