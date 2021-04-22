using Training.Metrics.Cpu;

namespace Training.Metrics.Factories
{
    public class CpuMetricFactory : IMetricFactory
    {
        public IMetric CreateClassificationAccuracy()
        {
            return new CpuClassificationAccuracy();
        }

        public IMetric CreateR2()
        {
            return new CpuR2();
        }

        public IMetric CreateMAE()
        {
            return new CpuMeanAbsoluteError();
        }
    }
}
