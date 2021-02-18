using System;
using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Model.WeightsInitializers
{
    public class XavierInitializer : IWeightsInitializer
    {
        public void InitWeights(IParameterizedLayer wLayer)
        {
            var n = wLayer.FIn + wLayer.FOut;
            var variance = 2.0 / n;
            float[] data = new float[wLayer.ParametersStorage.Weights.Size];
            for (int i = 0; i < wLayer.ParametersStorage.Weights.Size; i++)
            {
                data[i] = (float)RandomUtil.GetGaussian(0, Math.Sqrt(variance));
            }
            wLayer.ParametersStorage.Weights.Storage.Data = data;
        }
    }
}
