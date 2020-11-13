using System;

namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public class Relu : IFunction
    {
        public float Process(float x)
        {
            return x > 0 ? x : 0;
        }

        public float Derivative(float x)
        {
            return x > 0 ? 1 : 0;
        }
    }
}
