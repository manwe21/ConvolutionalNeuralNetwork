using System;
using System.Linq;
using Network.NeuralMath;

namespace Training.Metrics
{
    public class R2 : IMetric
    {
        public float Evaluate(Tensor real, Tensor predicted)
        {
            float res = 0;
            var chw = real.Channels * real.Height * real.Width;
            for (int b = 0; b < real.Batch; b++)
            {
                var meanT = real.Storage.Data.Average();
                float totalVariation = 0;
                float explainedVariation = 0;
                
                var start = chw * b;
                var fin = start + chw;
                for (int i = start; i < fin; i++)
                {
                    totalVariation += MathF.Pow(real[i] - meanT, 2);
                    explainedVariation += MathF.Pow(real[i] - predicted[i], 2);
                }
                res += 1 - explainedVariation / totalVariation;
            }

            return res / real.Batch;
        }
    }
}