using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Optimizers
{
    public interface IOptimizer    
    {
        public float LearningRate { get; }

        void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters, bool resetDw,
            int iteration);
    }
}
