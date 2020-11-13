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
            for (int i = 0; i < wLayer.Weights.Size; i++)
            {
                wLayer.Weights[i] = (float)RandomUtil.GetGaussian(_mean, _deviation);
            }
        }
    }
}
