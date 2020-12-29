using System;

namespace Network.NeuralMath.Functions.LossFunctions
{
    //also known as LogLoss
    public class CrossEntropy : ILossFunction
    {
        public float Process(Tensor output, Tensor correct)
        {
            var sum = 0.0f;
            for (int i = 0; i < output.Size; i++)
            {
                sum += correct[i] * MathF.Log(output[i] + Single.Epsilon);
            }
            return -sum;
        }

        //Possible division by 0 when network`s architecture is wrong
        //Single.Epsilon does not help (1 / Epsilon = +infinity)
        public float Derivative(float o, float t)
        {
            return -t/* / o*/;
        }
    }
}
