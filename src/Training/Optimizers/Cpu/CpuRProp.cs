using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers.Cpu
{
    public class CpuRProp : RProp
    {

        public CpuRProp(float learningRate) : base(learningRate)
        { }

        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var prevGradient = parameters["PrevGradient"];
            var delta = parameters["Delta"];
            var deltaW = parameters["DeltaWeight"];
            
            for (int i = 0; i < weights.Size; i++)
            {
                if (prevGradient[i] * gradients[i] > 0)
                {
                    delta[i] = MathF.Min(delta[i] * (float)EtaForward, 50);
                    deltaW[i] = -MathF.Sign(gradients[i]) * delta[i];
                    weights[i] = weights[i] + deltaW[i];
                    prevGradient[i] = gradients[i];
                }
                else if (prevGradient[i] * gradients[i] < 0)
                {
                    delta[i] = MathF.Max(delta[i] * (float)EtaBackward, 0e-6f);
                    prevGradient[i] = 0;
                }
                else
                {
                    delta[i] = -Math.Sign(gradients[i]) * delta[i];
                    weights[i] = weights[i] + deltaW[i];
                    prevGradient[i] = gradients[i];
                }
            }
        }

    }
}
