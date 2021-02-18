using Network.NeuralMath;

namespace Training.Metrics
{
    public interface IMetric
    {
        float Evaluate(Tensor real, Tensor predicted);
    }
}
