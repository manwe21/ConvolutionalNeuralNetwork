namespace Network.NeuralMath.Functions.LossFunctions
{
    public interface ILossFunction
    {    
        void Process(Tensor output, Tensor correct, Tensor loss);
        void Derivative(Tensor o, Tensor t, Tensor dy);
    }
}
