using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers.Cpu
{
    public class CpuGradientDescent : GradientDescent
    {
        public CpuGradientDescent(float learningRate) : base(learningRate)
        { }
        
        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            for (int i = 0; i < weights.Size; i++)
            {
                weights[i] -= LearningRate * gradients[i];
                
                if (resetDw)
                    gradients[i] = 0;
            }
        }

    }
}