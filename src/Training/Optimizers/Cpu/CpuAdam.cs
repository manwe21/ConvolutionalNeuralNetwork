using System;
using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Optimizers.Cpu
{
    public class CpuAdam : Adam
    {
        public CpuAdam(float learningRate) : base(learningRate) 
        { }
        
        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters, bool resetDw, int iteration)
        {
            var sTensor = parameters["S"];
            var dTensor = parameters["D"];
            
            for (int i = 0; i < weights.Size; i++)
            {
                sTensor[i] = Alpha * sTensor[i] + (1 - Alpha) * gradients[i];
                dTensor[i] = Beta * dTensor[i] + (1 - Beta) * gradients[i] * gradients[i];

                float s = sTensor[i] / (1 - MathF.Pow(Alpha, iteration));
                float d = dTensor[i] / (1 - MathF.Pow(Beta, iteration));

                weights[i] -= LearningRate / MathF.Sqrt(d + Single.Epsilon) * s;

                if(resetDw)
                    gradients[i] = 0;
            }
        }

    }
}
