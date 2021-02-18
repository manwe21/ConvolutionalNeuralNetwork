using System.Collections.Generic;
using Network.NeuralMath;

namespace Network.Model
{
    public class ParametersStorage
    {
        public Tensor Gradients { get; set; }
        public Tensor Weights { get; set; }
        public Tensor Biases { get; set; }
        public Dictionary<string, Tensor> Parameters { get; set; }
    }
}
