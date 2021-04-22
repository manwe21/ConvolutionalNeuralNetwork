using System;

namespace Network.NeuralMath.Functions.LossFunctions
{
    public class MeanSquaredError : ILossFunction, IGpuFunction
    {
        public string ForwardKernelName => "mean_squared_error";
        public string BackwardKernelName => "mean_squared_dy";
        
        public void Process(Tensor output, Tensor correct, Tensor loss)
        {
            var sizePerBatch = output.Size / output.Batch;
            for (int b = 0; b < output.Batch; b++)
            {
                var sum = 0.0f;
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    sum += MathF.Pow(output[i] - correct[i], 2);
                }

                loss[b] = sum / sizePerBatch;
            }
        }

        public void Derivative(Tensor o, Tensor t, Tensor dy)
        {
            int sizePerBatch = o.Size / o.Batch;
            for (int i = 0; i < o.Size; i++)
            {
                dy[i] = 2f / sizePerBatch * (o[i] - t[i]);
            }
        }
        
    }
}
