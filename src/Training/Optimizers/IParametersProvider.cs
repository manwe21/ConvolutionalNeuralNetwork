using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Optimizers
{
    public interface IParametersProvider
    {
        Dictionary<string, float> GetParameters();
    }
}