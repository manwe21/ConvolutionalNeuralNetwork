using System.Collections.Generic;
using Network.NeuralMath;

namespace Network.Model.Layers
{
    public interface IParameterizedLayer
    {
        ParametersStorage ParametersStorage { get; set; }
        int FIn { get; }
        int FOut { get; }

    }
}
