namespace Training.Metrics.Factories
{
    public interface IMetricFactory
    {
        IMetric CreateClassificationAccuracy();
        IMetric CreateR2();
        IMetric CreateMAE();
    }
}
