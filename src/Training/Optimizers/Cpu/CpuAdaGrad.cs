using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers.Cpu
{
    public class CpuAdaGrad : AdaGrad
    {
        public CpuAdaGrad(float learningRate) : base(learningRate)
        { }
        
        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var history = parameters["GradientHistory"];
            
            for (int i = 0; i < weights.Size; i++)
            {
                history[i] = history[i] + MathF.Pow(gradients[i], 2);
                weights[i] -= LearningRate / MathF.Sqrt(history[i] + float.Epsilon) * gradients[i];
            }
        }

    }
}