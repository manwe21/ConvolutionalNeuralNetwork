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

        public override void TrainModel(NeuralNetwork network)
        {
            base.TrainModel(network);
            
            var examplesPerEpoch = ExamplesCount / BatchSize;
            Stopwatch sw = Stopwatch.StartNew();
            
            for ( ; Epoch <= EpochsCount; Epoch++)
            {
                var examplesPassed = 0;
                Iteration = 1;
                double loss = 0;
                var correct = 0;
                
                foreach (var example in TrainingExamples)
                {
                    Network.Forward(example.Input);
                    Network.Output.LossDerivative(example.Output, LossFunction, Dy);
                    CalculateLoss(example.Output);
                    loss += Loss[0];
                    Network.Backward(Dy);
                    if (CheckModelOnCorrectOutput(example.Output))
                       correct++;
                    examplesPassed++;
                    
                    if (examplesPassed % BatchSize == 0)
                    {  
                        CorrectWeights();
                        var result = new IterationResult
                        {
                            Epoch = this.Epoch,
                            ExamplesPassed = examplesPassed,
                            Iteration = this.Iteration,
                            IterationTime = sw.Elapsed,
                            ExamplesPerEpoch = examplesPerEpoch,
                            Loss = loss / BatchSize,
                            Accuracy = (double) correct / BatchSize
                        };
                        RaiseIterationFinishedEvent(result);
                        loss = 0;
                        correct = 0;
                        Iteration++;
                        sw.Restart();
                    }
                }
                RaiseEpochFinishedEvent(new EpochResult());
            }
        }
    }
}
