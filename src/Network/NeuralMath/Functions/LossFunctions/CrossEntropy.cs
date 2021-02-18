using System;

namespace Network.NeuralMath.Functions.LossFunctions
{
    //also known as LogLoss
    public class CrossEntropy : ILossFunction
    {
        public void Process(Tensor output, Tensor correct, Tensor loss)
        {
            var sizePerBatch = output.Size / output.Batch;
            int count = 0;  
            for (int b = 0; b < output.Batch; b++) 
            {
                var sum = 0.0f;
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    sum += correct[count] * MathF.Log(output[count] + Single.Epsilon);
                    count++;
                }

                loss[b, 0, 0, 0] = -sum;
            }

        }

        //Possible division by 0 when network`s architecture is wrong
        //Single.Epsilon does not help (1 / Epsilon = +infinity)
        //
        //Solution - if cross entropy is used with softmax, o[i] will be reduced
        public void Derivative(Tensor o, Tensor t, Tensor dy)
        {
            for (int i = 0; i < o.Size; i++)
            {
                dy[i] = -t[i] / o[i];
            }
        }
    }
}
