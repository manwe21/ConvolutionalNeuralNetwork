using System;
using Network.NeuralMath.Cpu;
using Network.NeuralMath.Gpu;

namespace Network.NeuralMath
{
    public abstract class TensorBuilder
    {
        public abstract Tensor Empty();
        public abstract Tensor OfStorage(TensorStorage storage);
        public abstract Tensor OfShape(Shape shape);
        public abstract Tensor Filled(Shape shape, float value);
        public abstract Tensor Filled(Shape shape, Func<float> func);

        public static TensorBuilder OfType(Type tensorType)
        {
            if (tensorType == typeof(CpuTensor))
                return new CpuBuilder();
            if(tensorType == typeof(GpuTensor))
                return new GpuBuilder();
            return null;
        }

        public static TensorBuilder Create()
        {
            if (Global.ComputationType == ComputationType.Cpu)
                return new CpuBuilder();
            return new GpuBuilder();
        }
        
    }
}
