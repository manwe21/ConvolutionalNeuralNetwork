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
            for (int i = 0; i < wLayer.Weights.Size; i++)
            {
                wLayer.Weights[i] = (float)_value;
            }
        }
    }
}
