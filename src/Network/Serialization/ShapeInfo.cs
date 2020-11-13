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

        public ShapeInfo(Shape shape)
        {
            B = shape.Dimensions[0];
            C = shape.Dimensions[1];
            H = shape.Dimensions[2];
            W = shape.Dimensions[3];
        }
        
    }
}
