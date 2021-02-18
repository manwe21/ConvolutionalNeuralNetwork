using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Network.Model;
using Training.Data;
using Training.Trainers.Settings;

namespace Training.Trainers
{
    public class MiniBatchTrainer : BaseTrainer
    {
        public int BatchSize { get; }

        public MiniBatchTrainer(IExamplesSource examplesSource, MiniBatchTrainerSettings settings)
            : base(examplesSource, settings)
        {
            BatchSize = settings.BatchSize;
        }
        
        public MiniBatchTrainer(IEnumerable<Example> examples, MiniBatchTrainerSettings settings)
            : base(examples, settings)
        {
            BatchSize = settings.BatchSize;
        }

        public override void TrainModel(NeuralLayeredNetwork network)
        {
            base.TrainModel(network);
            Stopwatch sw = Stopwatch.StartNew();
            for ( ; Epoch <= EpochsCount; Epoch++)
            {
                Iteration = 1;
                foreach (var example in TrainingExamples)
                {
                    Network.Forward(example.Input);
                    CalculateLoss(example.Output);
                    Network.Output.LossDerivative(example.Output, LossFunction, Dy);
                    Network.Backward(Dy);
                    CorrectWeights();
                    
                    var result = new IterationResult
                    {
                        Epoch = this.Epoch,
                        Iteration = this.Iteration,
                        IterationTime = sw.Elapsed,
                        ExamplesPerEpoch = ExamplesCount,
                        Loss = Loss.Storage.Data.Average(),
                        Accuracy = 0,//Metric.Evaluate(example.Output, Network.Output),
                        EpochsCount = this.EpochsCount
                    };
                    RaiseIterationFinishedEvent(result);
                    Iteration++;
                    sw.Restart();
                }
                
                RaiseEpochFinishedEvent(new EpochResult());
            }
        }
    }
}
