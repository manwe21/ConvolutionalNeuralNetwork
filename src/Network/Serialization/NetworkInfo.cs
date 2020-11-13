using System;
using System.Collections.Generic;
using Network.NeuralMath;

namespace Network.Serialization
{
    [Serializable]
    public class NetworkInfo
    {
        public ShapeInfo InputShape { get; set; }
        public List<LayerInfo> LayersInfo { get; set; } = new List<LayerInfo>();
    }
}
