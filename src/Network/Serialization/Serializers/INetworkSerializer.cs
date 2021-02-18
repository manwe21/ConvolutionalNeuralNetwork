using System.IO;
using Network.Model;

namespace Network.Serialization.Serializers
{
    public interface INetworkSerializer
    {
        void Serialize(NeuralLayeredNetwork network, string filePath);
        void Serialize(NeuralLayeredNetwork network, Stream stream);
        
        NeuralLayeredNetwork Deserialize(string filePath);
        NeuralLayeredNetwork Deserialize(Stream stream);    
    }
}
