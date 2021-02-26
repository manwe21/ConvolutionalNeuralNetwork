using System;
using System.Collections.Generic;
using System.Linq;
using Network.Model.Exceptions;
using Network.Model.Layers;
using Network.NeuralMath;
using Network.Serialization;
using Network.Serialization.Serializers;

namespace Network.Model
{
    public class NeuralLayeredNetwork : INetwork
    {
        private readonly List<BaseLayer> _layers;
        public int LayersCount => _layers.Count;
        
        public IReadOnlyList<BaseLayer> Layers => _layers;
        
        public Shape InputShape { get; }
        
        public Tensor Output => _layers[LayersCount - 1].Output;

        public List<IParameterizedLayer> ParameterizedLayers
        {
            get
            {
                return _layers.Where(l => l is IParameterizedLayer).Cast<IParameterizedLayer>().ToList();
            }
        }

        public IEnumerable<ParametersStorage> GetParameters()
        {
            for (int i = 0; i < LayersCount; i++)
            {
                if (_layers[i] is IParameterizedLayer pLayer)
                {
                    yield return pLayer.ParametersStorage;
                }
            }
        }

        public NeuralLayeredNetwork(Shape inputShape)
        {
            InputShape = inputShape;
            _layers = new List<BaseLayer>();
        }

        public Tensor Forward(Tensor input)
        {
            if (!_layers.All(l => l.IsInit))
                throw new ModelIsNotInitializedException();
            
            if (input.Storage.Shape != InputShape)
                throw new ArgumentException("Input tensor has incompatible shape");
            
            var tensor = _layers[0].Forward(input);
            for (int i = 1; i < LayersCount; i++)
            {
                tensor = _layers[i].Forward(tensor);
            }

            return Output;
        }
        
        public void Backward(Tensor dy)
        {
            var tensor = _layers[LayersCount - 1].Backward(dy);
            for (int i = LayersCount - 2; i >= 0; i--)
            {
                tensor = _layers[i].Backward(tensor);
            }
        }

        public void AddLayer(BaseLayer layer)
        {
            Shape inputShape;
            if (Layers.Any())
            {
                var lastLayer = _layers.Last();
                inputShape = lastLayer.OutputShape;
                layer.Prev = lastLayer;
                lastLayer.Next = layer;
            }
            else inputShape = InputShape;
            
            if (!layer.IsInit)
                layer.Initialize(inputShape);
            _layers.Add(layer);
        }

        public NetworkInfo GetNetworkInfo()
        {
            var info = new NetworkInfo
            {
                InputShape = new ShapeInfo(this.InputShape)
            };

            foreach (var layer in _layers)
            {
                info.LayersInfo.Add(layer.GetLayerInfo());
            }
            return info;
        }

    }
}
