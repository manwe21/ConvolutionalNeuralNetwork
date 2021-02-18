using System;
using System.IO;
using System.Text.Json;
using Network.Model;
using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Serialization.Serializers
{
    public class JsonNetworkSerializer : INetworkSerializer
    {
        public void Serialize(NeuralLayeredNetwork network, string filePath)
        {
            FileInfo file = new FileInfo(filePath);
            if (file.Extension != ".json")
                throw new ArgumentException($"File {filePath} has wrong format");
            
            using var stream = new FileStream(filePath, FileMode.Create);
            Serialize(network, stream);
        }

        public void Serialize(NeuralLayeredNetwork network, Stream stream)
        {
            var data = network.GetNetworkInfo();
            using TextWriter writer = new StreamWriter(stream);
            var options = new JsonSerializerOptions
            {
                Converters = { new LayerInfoConverter() },
                WriteIndented = true
            };
            var json = JsonSerializer.Serialize(data, options);
            writer.Write(json);
        }

        public NeuralLayeredNetwork Deserialize(string filePath)
        {
            FileInfo file = new FileInfo(filePath);
            if (!file.Exists)
                throw new ArgumentException($"File {filePath} does not exists");
            if (file.Extension != ".json")
                throw new ArgumentException($"File {filePath} has wrong format");
            
            NetworkInfo info;
            using (var stream = new FileStream(filePath, FileMode.Open))
            {
                using var reader = new StreamReader(stream);
                var json = reader.ReadToEnd();
                var options = new JsonSerializerOptions
                {
                    Converters = { new LayerInfoConverter() },
                    WriteIndented = true
                };

                info = JsonSerializer.Deserialize<NetworkInfo>(json, options);
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
            using var reader = new StreamReader(stream);
            var json = reader.ReadToEnd();
            var options = new JsonSerializerOptions
            {
                Converters = { new LayerInfoConverter() },
                WriteIndented = true
            };

            var info = JsonSerializer.Deserialize<NetworkInfo>(json, options);
            
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
