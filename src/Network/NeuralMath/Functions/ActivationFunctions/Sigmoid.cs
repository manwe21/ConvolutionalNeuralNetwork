namespace Network.NeuralMath.Functions.ActivationFunctions
{
    public class Sigmoid : IFunction
    {
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
