using Network.NeuralMath;
using Network.NeuralMath.Cpu;

namespace CpuTensorTests
{
    public class CpuTensorTests : TensorTests
    {
        protected override Tensor CreateTensor(Shape shape, float[] data)
        {
            Tensor t = new CpuTensor(new CpuStorage(shape));
            t.Storage.SetData(data);
            return t;
        }

        protected override Tensor CreateTensor()
        {
            return CpuTensor.Build.Empty();
        }
    }
}
