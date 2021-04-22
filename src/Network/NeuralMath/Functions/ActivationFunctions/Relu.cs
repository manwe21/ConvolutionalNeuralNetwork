using System;

namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public class Relu : IFunction, IGpuFunction
    {
        public string ForwardKernelName => "relu_forward";
        public string BackwardKernelName => "relu_backward";
        
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
