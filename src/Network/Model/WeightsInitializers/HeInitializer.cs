using System;
using System.Diagnostics;
using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Model.WeightsInitializers
{
    public class HeInitializer : IWeightsInitializer
    {
        public void InitWeights(IParameterizedLayer wLayer)
        {   
            var variance = 2.0 / wLayer.FIn;
            float[] data = new float[wLayer.Weights.Size];
            for (int i = 0; i < wLayer.Weights.Size; i++)
            {
                data[i] = (float)RandomUtil.GetGaussian(0, Math.Sqrt(variance));
            }
            
            wLayer.Weights.Storage.SetData(data);
        }
    }
}
