namespace Training.Optimizers.Factories
{
    public interface IOptimizerFactory
    {
        IOptimizer CreateAdam(float learningRate);
        IOptimizer CreateAdaDelta(float learningRate);
        IOptimizer CreateGradientDescent(float learningRate);
        IOptimizer CreateAdaGrad(float learningRate);
        IOptimizer CreateRProp(float learningRate);
    }
}
