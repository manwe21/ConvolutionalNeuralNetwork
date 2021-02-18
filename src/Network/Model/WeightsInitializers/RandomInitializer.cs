using Network.Model.Layers;
using Network.NeuralMath;

namespace Network.Model.WeightsInitializers
{
    public class RandomInitializer : IWeightsInitializer
    {
        private readonly double _minValue;
        private readonly double _maxValue;

        public RandomInitializer(double minValue, double maxValue)
        {
            _minValue = minValue;
            _maxValue = maxValue;
        }
        
        public void InitWeights(IParameterizedLayer wLayer)
        {
            float[] data = new float[wLayer.ParametersStorage.Weights.Size];
            for (int i = 0; i < wLayer.ParametersStorage.Weights.Size; i++)
            {
                wLayer.ParametersStorage.Weights[i] = (float)RandomUtil.GetRandomNumber(_minValue, _maxValue);
            }
            wLayer.ParametersStorage.Weights.Storage.Data = data;
        }
        
    }
}
