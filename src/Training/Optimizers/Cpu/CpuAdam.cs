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
            
            fixed (float* wPtr = weights.Storage.Array)
            {
                fixed (float* sPtr = sTensor.Storage.Array)
                {
                    fixed (float* dPtr = dTensor.Storage.Array)
                    {
                        fixed (float* gPtr = gradients.Storage.Array)
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
