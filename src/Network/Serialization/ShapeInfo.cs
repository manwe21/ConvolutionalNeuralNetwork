using System;
using Network.NeuralMath;

namespace Network.Serialization
{
    [Serializable]
    public class ShapeInfo
    {
        public int B { get; set; }
        public int C { get; set; }
        public int H { get; set; }
        public int W { get; set; }

        public ShapeInfo()
        {
            
        }

        public ShapeInfo(Shape shape)
        {
            B = shape[0];
            C = shape[1];
            H = shape[2];
            W = shape[3];
        }
        
    }
}
