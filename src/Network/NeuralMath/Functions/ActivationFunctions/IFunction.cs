using System;

namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public interface IFunction
    {
        float Process(float x);
        float Derivative(float x);
    }
}
