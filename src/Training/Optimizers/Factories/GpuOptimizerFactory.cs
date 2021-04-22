using System;
using Training.Optimizers.Gpu;

namespace Training.Optimizers.Factories
{
    public class GpuOptimizerFactory : IOptimizerFactory
    {
        public IOptimizer CreateAdam(float learningRate)
        {
            return new GpuAdam(learningRate);
        }

        public IOptimizer CreateAdaDelta(float learningRate)
        {
            return new GpuAdaDelta(learningRate);
        }

        public IOptimizer CreateGradientDescent(float learningRate)
        {
            return new GpuGradientDescent(learningRate);
        }

        public IOptimizer CreateAdaGrad(float learningRate)
        {
            return new GpuAdaGrad(learningRate);
        }

        public IOptimizer CreateRProp(float learningRate)
        {
            throw new NotImplementedException();
        }
    }
}
