using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Optimizers
{
    public abstract class RProp : IOptimizer, IParametersProvider
    {
        protected const double EtaForward = 1.2;
        protected const double EtaBackward = 0.5;
        
        protected RProp(float learningRate)
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
                { "Delta", 0.1f },
                { "DeltaWeight", 0.1f },
                { "PrevGradient", 0 }
            };
        }
    }
}
