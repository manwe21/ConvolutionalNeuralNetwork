using System;

namespace Network.NeuralMath.Functions.LossFunctions
{
    public class MeanSquaredError : ILossFunction
    {
        public float Process(Tensor output, Tensor correct)
        {
            var sum = 0.0f;
            for (int i = 0; i < output.Size; i++)
            {
                sum += MathF.Pow(output[i] - correct[i], 2);
            }
            return sum / output.Size;
        }

        public float Derivative(float o, float t)
        {
            return o - t;
        }
    }
}