using Network.Model.Layers;

namespace Network.Model.WeightsInitializers
{
    public interface IWeightsInitializer
    {
        public void InitWeights(IParameterizedLayer wLayer);
    }
}
