using System.Collections.Generic;
using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Model
{
    public interface INetwork
    {
        Tensor Forward(Tensor x);
        void Backward(Tensor dy);   
        Tensor Output { get; }

        IEnumerable<ParametersStorage> GetParameters();
    }
}
