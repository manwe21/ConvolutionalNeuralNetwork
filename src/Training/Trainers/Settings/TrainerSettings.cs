using Network.NeuralMath.Functions.LossFunctions;
using Training.Metrics;
using Training.Metrics.Cpu;
using Training.Optimizers;
using Training.Optimizers.Cpu;

namespace Training.Trainers.Settings
{
    public class TrainerSettings
    {
        public int EpochsCount { get; set; }
        public IOptimizer Optimizer { get; set; }
        public ILossFunction LossFunction { get; set; }
        public IMetric Metric { get; set; }

        public TrainerSettings()
        {
            EpochsCount = 1;
            Optimizer = new CpuAdam(1e-3f);
            LossFunction = new CrossEntropy();
            Metric = new CpuClassificationAccuracy();
        }
    }
}
