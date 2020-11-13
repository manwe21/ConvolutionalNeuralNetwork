using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers
{
    public abstract class Adam : IOptimizer, IParametersProvider
    {
        protected const float Alpha = 0.9f;
        protected const float Beta = 0.999f;

        public Adam(float learningRate)
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
                { "S", 0 },
                { "D", 0 }
            };
        }
    }
}
