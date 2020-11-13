using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers.Cpu
{
    public class CpuAdaDelta : AdaDelta
    {
        public CpuAdaDelta(float learningRate) : base(learningRate)
        { }
        
        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var esq = parameters["Esq"];
            
            for (int i = 0; i < weights.Size; i++)
            {
                esq[i] = Gamma * esq[i] + (1 - Gamma) * MathF.Pow(gradients[i], 2);
                weights[i] -= LearningRate / MathF.Sqrt(esq[i] + Single.Epsilon) * gradients[i];
            }
        }
    }
}
