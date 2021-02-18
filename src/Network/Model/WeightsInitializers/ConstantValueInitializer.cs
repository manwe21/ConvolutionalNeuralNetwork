using Network.Model.Layers;

namespace Network.Model.WeightsInitializers
{
    public class ConstantValueInitializer : IWeightsInitializer
    {
        private readonly double _value;

        public ConstantValueInitializer(double value)
        {
            _value = value;
        }

        public void InitWeights(IParameterizedLayer wLayer)
        {
            float[] data = new float[wLayer.ParametersStorage.Weights.Size];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)_value;
            }

            wLayer.ParametersStorage.Weights.Storage.Data = data;
        }
    }
}
