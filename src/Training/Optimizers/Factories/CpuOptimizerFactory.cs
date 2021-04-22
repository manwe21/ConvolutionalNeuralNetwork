using Training.Optimizers.Cpu;

namespace Training.Optimizers.Factories
{
    public class CpuOptimizerFactory : IOptimizerFactory
    {
        public IOptimizer CreateAdam(float learningRate)
        {
            return new CpuAdam(learningRate);
        }

        public IOptimizer CreateAdaDelta(float learningRate)
        {
            return new CpuAdaDelta(learningRate);
        }

        public IOptimizer CreateGradientDescent(float learningRate)
        {
            return new CpuGradientDescent(learningRate);
        }

        public IOptimizer CreateAdaGrad(float learningRate)
        {
            return new CpuAdaGrad(learningRate);
        }

        public IOptimizer CreateRProp(float learningRate)
        {
            return new CpuRProp(learningRate);
        }
    }
}
