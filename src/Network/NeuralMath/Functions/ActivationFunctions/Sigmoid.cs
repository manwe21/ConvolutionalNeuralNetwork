namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public class Sigmoid : IFunction, IGpuFunction
    {
        public string ForwardKernelName => "sigmoid_forward";
        public string BackwardKernelName => "sigmoid_backward";
        
        public float Process(float x)
        {
            return 1 / (1 + System.MathF.Exp(-x));
        }
            
        public float Derivative(float x)
        {
            return Process(x) * (1 - Process(x));
        }

    }
}
