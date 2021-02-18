using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Network.Model;
using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Serialization.Serializers
{
    //todo refactor
    public class BinaryNetworkSerializer : INetworkSerializer
    {
        public void Serialize(NeuralLayeredNetwork network, string filePath)
        {
            FileInfo file = new FileInfo(filePath);
            if (file.Extension != ".cnn")
                throw new ArgumentException($"File {filePath} has wrong format");

            var data = network.GetNetworkInfo();
            BinaryFormatter formatter = new BinaryFormatter();

            using var stream = new FileStream(filePath, FileMode.OpenOrCreate);
            formatter.Serialize(stream, data);
        }

        public void Serialize(NeuralLayeredNetwork network, Stream stream)
        {
            var data = network.GetNetworkInfo();
            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, data);
        }

        public NeuralLayeredNetwork Deserialize(string filePath)
        {
            FileInfo file = new FileInfo(filePath);
            if (!file.Exists)
                throw new ArgumentException($"File {filePath} does not exists");
            if (file.Extension != ".cnn")
                throw new ArgumentException($"File {filePath} has wrong format");

            NetworkInfo info;
            BinaryFormatter formatter = new BinaryFormatter();
            using (var stream = new FileStream(filePath, FileMode.OpenOrCreate))
            {
                info = (NetworkInfo)formatter.Deserialize(stream);
            }
            var inputShape = new Shape(info.InputShape.B, info.InputShape.C, info.InputShape.H, info.InputShape.W);
            var network = new NeuralLayeredNetwork(inputShape);
            foreach (var layerInfo in info.LayersInfo)
            {
                var layerType = Type.GetType(layerInfo.LayerType);
                if(layerType is null)
                    throw new ArgumentException();
                var layer = (BaseLayer) Activator.CreateInstance(layerType, layerInfo);
                network.AddLayer(layer);
            }
            return network;
        }

        public NeuralLayeredNetwork Deserialize(Stream stream)
        {
            NetworkInfo info;
            BinaryFormatter formatter = new BinaryFormatter();
            info = (NetworkInfo)formatter.Deserialize(stream);
            var inputShape = new Shape(info.InputShape.B, info.InputShape.C, info.InputShape.H, info.InputShape.W);
            var network = new NeuralLayeredNetwork(inputShape);
            foreach (var layerInfo in info.LayersInfo)
            {
                var layerType = Type.GetType(layerInfo.LayerType);
                if(layerType is null)
                    throw new ArgumentException();
                var layer = (BaseLayer) Activator.CreateInstance(layerType, layerInfo);
                network.AddLayer(layer);
            }
            return network;
        }
    }
}