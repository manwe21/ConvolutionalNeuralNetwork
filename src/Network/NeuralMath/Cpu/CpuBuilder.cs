using System;

namespace Network.NeuralMath.Cpu
{
    public class CpuBuilder : TensorBuilder
    {
        public override Tensor Empty()
        {
            return new CpuTensor();
        }

        public override Tensor OfStorage(TensorStorage storage)
        {
            if(!(storage is CpuStorage cpuStorage))
                throw new NotImplementedException();
            return new CpuTensor(cpuStorage);    
        }

        public override Tensor OfShape(Shape shape)
        {
            return new CpuTensor(new CpuStorage(shape));
        }

        public override Tensor Filled(Shape shape, float value)
        {
            var tensor = new CpuTensor(new CpuStorage(shape));
            for (int i = 0; i < tensor.Size; i++)
            {
                tensor[i] = value;
            }

            return tensor;
        }

        public override Tensor Filled(Shape shape, Func<float> func)
        {
            var tensor = new CpuTensor(new CpuStorage(shape));
            for (int i = 0; i < tensor.Size; i++)
            {
                tensor[i] = func();
            }
    
            return tensor;
        }
    }
}