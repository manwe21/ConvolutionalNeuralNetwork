using Network.NeuralMath;
using Network.NeuralMath.Gpu;

namespace CpuTensorTests
{
    public class GpuTensorTests : TensorTests
    {
        protected override Tensor CreateTensor(Shape shape, float[] data)
        {
            Tensor t = GpuTensor.Build.OfStorage(new GpuStorage(shape, data));
            return t;
        }

        protected override Tensor CreateTensor()
        {
            return GpuTensor.Build.Empty();
        }
    }
}
