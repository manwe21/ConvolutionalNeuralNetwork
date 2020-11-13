using System.Collections.Generic;
using Network.NeuralMath;

namespace Network.Model.Layers
{
    public interface IParameterizedLayer
    {
        Tensor Weights { get; }       
        Tensor Biases { get; }    
        Tensor WeightsGradient { get; }
        int FIn { get; }
        int FOut { get; }
        Dictionary<string, Tensor> Parameters { get; set; }
    }
}
