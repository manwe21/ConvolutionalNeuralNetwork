using Network.NeuralMath;
using Network.NeuralMath.Cpu;

namespace CpuTensorTests
{
    public class CpuTensorTests : TensorTests
    {
        protected override Tensor CreateTensor(Shape shape, float[] data)
        {
            Tensor t = CpuTensor.Build.OfStorage(new CpuStorage(shape, data));
            return t;
        }

        protected override Tensor CreateTensor()
        {
            return CpuTensor.Build.Empty();
        }
    }
}
