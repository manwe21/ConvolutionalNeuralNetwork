using System;

namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public class Tanh : IFunction, IGpuFunction
    {
        public string ForwardKernelName => "tanh_forward";
        public string BackwardKernelName => "tanh_backward";
        
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
