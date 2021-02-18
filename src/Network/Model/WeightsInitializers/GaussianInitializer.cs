using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Model.WeightsInitializers
{
    public class GaussianInitializer : IWeightsInitializer
    {
        private readonly double _mean;
        private readonly double _deviation;

        public GaussianInitializer(double mean, double deviation)
        {
            _mean = mean;
            _deviation = deviation;
        }

        public void InitWeights(IParameterizedLayer wLayer)
        {
            float[] data = new float[wLayer.ParametersStorage.Weights.Size];
            for (int i = 0; i < wLayer.ParametersStorage.Weights.Size; i++)
            {
                data[i] = (float)RandomUtil.GetGaussian(_mean, _deviation);
            }
            wLayer.ParametersStorage.Weights.Storage.Data = data;
        }
    }
}
