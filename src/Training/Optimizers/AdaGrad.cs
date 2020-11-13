using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers    
{
    public abstract class AdaGrad : IOptimizer, IParametersProvider
    {
        protected AdaGrad(float learningRate)
        {
            LearningRate = learningRate;
        }

        public float LearningRate { get; }
        
        public abstract void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration);

        public Dictionary<string, float> GetParameters()
        {
            return new Dictionary<string, float>
            {
                { "GradientHistory", 0 }
            };
        }

    }
}