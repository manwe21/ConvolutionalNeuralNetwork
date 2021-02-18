using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Optimizers
{
    public abstract class AdaDelta : IOptimizer, IParametersProvider
    {
        protected const float Gamma = 0.4f;
        
        public float LearningRate { get; }

        protected AdaDelta(float learningRate)
        {
            LearningRate = learningRate;
        }

        public abstract void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration);

        public Dictionary<string, float> GetParameters()
        {
            return new Dictionary<string, float>
            {
                { "Esq", 0}
            };
        }
    }
}
