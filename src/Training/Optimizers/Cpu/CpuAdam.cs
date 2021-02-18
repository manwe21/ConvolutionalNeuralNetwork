using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Training.Trainers;

namespace Training.Optimizers.Cpu
{
    public class CpuAdam : Adam
    {
        public CpuAdam(float learningRate) : base(learningRate) 
        { }
        
        public override unsafe void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var sTensor = parameters["S"];
            var dTensor = parameters["D"];
            
            /*for (int i = 0; i < weights.Size; i++)
            {
                sTensor[i] = Alpha * sTensor[i] + (1 - Alpha) * gradients[i];
                dTensor[i] = Beta * dTensor[i] + (1 - Beta) * gradients[i] * gradients[i];

                float s = sTensor[i] / (1 - MathF.Pow(Alpha, iteration));
                float d = dTensor[i] / (1 - MathF.Pow(Beta, iteration));

                weights[i] -= LearningRate / MathF.Sqrt(d + Single.Epsilon) * s;

                if(resetDw)
                    gradients[i] = 0;

            }*/
            
            fixed (float* wPtr = weights.Storage.Data)
            {
                fixed (float* sPtr = sTensor.Storage.Data)
                {
                    fixed (float* dPtr = dTensor.Storage.Data)
                    {
                        fixed (float* gPtr = gradients.Storage.Data)
                        {
                            for (int i = 0; i < weights.Size; i++)
                            {
                                *(sPtr + i) = Alpha * *(sPtr + i) + (1 - Alpha) * *(gPtr + i);
                                *(dPtr + i) = Beta * *(dPtr + i) + (1 - Beta) * *(gPtr + i) * *(gPtr + i);

                                float s = *(sPtr + i) / (1 - MathF.Pow(Alpha, iteration));
                                float d = *(dPtr + i) / (1 - MathF.Pow(Beta, iteration));

                                *(wPtr + i) -= LearningRate / MathF.Sqrt(d + Single.Epsilon) * s;

                                if(resetDw)
                                    *(gPtr + i) = 0;

                            }
                        }
                    }
                }
            }

        }

    }
}
