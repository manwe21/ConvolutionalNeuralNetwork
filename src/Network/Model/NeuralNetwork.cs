using System;
using System.Collections.Generic;
using System.Linq;
using Network.Model.Layers;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model
{
    public class NeuralNetwork : INetwork
    {
        public List<BaseLayer> Layers { get; }
        public int LayersCount => Layers.Count;
        
        public Shape InputShape { get; }
        
        public Tensor Output => Layers[LayersCount - 1].Output;

        public List<IParameterizedLayer> ParameterizedLayers
        {
            get
            {
                return Layers.Where(l => l is IParameterizedLayer).Cast<IParameterizedLayer>().ToList();
            }
        }


        public NeuralNetwork(Shape inputShape)
        {
            InputShape = inputShape;
            Layers = new List<BaseLayer>();
        }

        public Tensor Forward(Tensor input)
        {
            if(!Layers.All(l => l.IsInit))
                throw new Exception("Model is not initialize");
            
            var tensor = Layers[0].Forward(input);
            for (int i = 1; i < LayersCount; i++)
            {
                tensor = Layers[i].Forward(tensor);
            }

            return Output;
        }
        
        public void Backward(Tensor dy)
        {
            var tensor = Layers[LayersCount - 1].Backward(dy);
            for (int i = LayersCount - 2; i >= 0; i--)
            {
                tensor = Layers[i].Backward(tensor);
            }
        }

        public void AddLayer(BaseLayer layer)
        {
            Shape inputShape;
            if (LayersCount > 0)
            {
                inputShape = Layers.Last().OutputShape;
                layer.Prev = Layers.Last();
                Layers.Last().Next = layer;
            }
            else inputShape = InputShape;
            
            if (!layer.IsInit)
                layer.Initialize(inputShape);
            Layers.Add(layer);
        }

        public void Save(string filePath)
        {
            NetworkSerializer.SerializeNetwork(this, filePath);
        }

        public static NeuralNetwork Load(string filePath)
        {
            return NetworkSerializer.DeserializeNetwork(filePath);
        }

        public NetworkInfo GetNetworkInfo()
        {
            var info = new NetworkInfo
            {
                InputShape = new ShapeInfo(this.InputShape)
            };

            foreach (var layer in Layers)
            {
                info.LayersInfo.Add(layer.GetLayerInfo());
            }
            return info;
        }

    }
}
