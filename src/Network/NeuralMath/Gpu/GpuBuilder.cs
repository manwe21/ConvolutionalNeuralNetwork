using System;

namespace Network.NeuralMath.Gpu
{
    public class GpuBuilder : TensorBuilder
    {
        public override Tensor Empty()
        {
            return new GpuTensor();
        }

        public override Tensor OfStorage(TensorStorage storage)
        {
            return new GpuTensor((GpuStorage)storage);
        }

        public override Tensor OfShape(Shape shape)
        {
            return new GpuTensor(new GpuStorage(shape));
        }

        public override Tensor Filled(Shape shape, float value)
        {
            var tensor = new GpuTensor(new GpuStorage(shape));
            tensor.Fill(value);
            return tensor;
        }

        public override Tensor Filled(Shape shape, Func<float> func)
        {
            throw new NotImplementedException();
        }
    }
}
