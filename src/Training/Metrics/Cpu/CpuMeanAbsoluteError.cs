using System;
using Network.NeuralMath;

namespace Training.Metrics.Cpu
{
    public class CpuMeanAbsoluteError : IMetric
    {
        public float Evaluate(Tensor real, Tensor predicted)
        {
            float res = 0;
            var chw = real.Channels * real.Height * real.Width;
            for (int b = 0; b < real.Batch; b++)
            {
                var sum = 0f;
                var start = chw * b;
                var fin = start + chw;
                for (int i = start; i < fin; i++)
                {
                    sum += Math.Abs(predicted[i] - real[i]);
                }

                res += sum / chw;
            }

            return res / real.Batch;
        }
    }
}
