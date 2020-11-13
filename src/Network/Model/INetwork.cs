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
        
        //TODO Need to change this for work with more abstract network
        List<IParameterizedLayer> ParameterizedLayers { get; }
    }
} 
