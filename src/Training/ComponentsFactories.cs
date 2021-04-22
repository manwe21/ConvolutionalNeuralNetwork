using Network;
using Training.Metrics.Factories;
using Training.Optimizers.Factories;

namespace Training
{
    public static class ComponentsFactories
    {
        public static IOptimizerFactory OptimizerFactory
        {
            get
            {
                if (Global.ComputationType == ComputationType.Cpu) return new CpuOptimizerFactory();
                return new GpuOptimizerFactory();
            }
        }
        
        public static IMetricFactory MetricFactory
        {
            get
            {
                if (Global.ComputationType == ComputationType.Cpu) return new CpuMetricFactory();
                return new GpuMetricFactory();
            }
        }
    }
}
