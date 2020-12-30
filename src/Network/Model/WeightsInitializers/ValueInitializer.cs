using Network.Model.Layers;

namespace Network.Model.WeightsInitializers
{
    public class ValueInitializer : IWeightsInitializer
    {
        private readonly double _value;

        public ValueInitializer(double value)
        {
            _value = value;
        }

        public void InitWeights(IParameterizedLayer wLayer)
        {
            for (int i = 0; i < wLayer.ParametersStorage.Weights.Size; i++)
            {
                wLayer.ParametersStorage.Weights[i] = (float)_value;
            }
        }
    }
}
