using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;
using Network.Model;
using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Serialization
{
    public static class NetworkSerializer
    {
        public static void SerializeNetwork(NeuralNetwork network, string filePath)
        {
            FileInfo file = new FileInfo(filePath);
            if (file.Extension != ".cnn")
                throw new ArgumentException($"File {filePath} has wrong format");

            var data = network.GetNetworkInfo();
            BinaryFormatter formatter = new BinaryFormatter();

            using var stream = new FileStream(filePath, FileMode.OpenOrCreate);
            formatter.Serialize(stream, data);
        }

        public static NeuralNetwork DeserializeNetwork(string filePath)
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
            var network = new NeuralNetwork(inputShape);
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
