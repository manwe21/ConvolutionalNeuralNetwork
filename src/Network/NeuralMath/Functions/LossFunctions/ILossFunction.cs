namespace Network.NeuralMath.Functions.LossFunctions
{
    public interface ILossFunction
    {    
        float Process(Tensor output, Tensor correct);
        float Derivative(float o, float t);
    }
}    
