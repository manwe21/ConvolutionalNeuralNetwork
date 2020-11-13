using System;

namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public class Tanh : IFunction
    {
        public float Process(float x)
        {
            return (MathF.Exp(x) - MathF.Exp(-x)) / (MathF.Exp(x) + MathF.Exp(-x));
        }

        public float Derivative(float x)
        {
            return 1 - MathF.Pow(Process(x), 2);
        }
    }
}
