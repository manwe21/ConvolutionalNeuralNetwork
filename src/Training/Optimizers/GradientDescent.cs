using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Optimizers
{
    public abstract class GradientDescent : IOptimizer
    {
        protected GradientDescent(float learningRate)
        {
            LearningRate = learningRate;
        }

        public float LearningRate { get; }
        
        public abstract void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration);
    }
}
