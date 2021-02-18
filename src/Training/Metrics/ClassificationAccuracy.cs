using System;
using Network.NeuralMath;

namespace Training.Metrics
{
    public class ClassificationAccuracy : IMetric
    {
        public float Evaluate(Tensor real, Tensor predicted)
        {
            var total = real.Batch;
            var correct = 0;

            var chw = real.Channels * real.Height * real.Width;
            for (var b = 0; b < real.Batch; b++)
            {
                var max = Single.MinValue;  
                var maxI = 0;
                var max2 = Single.MinValue;
                var max2I = 0;

                var start = chw * b;
                var fin = start + chw;
                for (var i = start; i < fin; i++)
                {
                    if (real[i] > max)
                    {
                        maxI = i;
                        max = real[i];
                    }

                    if (predicted[i] > max2)
                    {
                        max2I = i;
                        max2 = predicted[i];
                    }
                }

                if (maxI == max2I)
                    correct++;
            }

            return (float)correct / total;
        }
    }
}
