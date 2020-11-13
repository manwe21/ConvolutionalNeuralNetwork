using System;
using System.Collections.Generic;
using System.Diagnostics;
using Network.Model;
using Training.Data;
using Training.Trainers.Settings;

namespace Training.Trainers
{
    [Obsolete("Use mini-batch trainer with [BatchSize = 1] instead")]
    public class OnlineTrainer : BaseTrainer
    {
        public OnlineTrainer(IExamplesSource examplesSource, TrainerSettings settings) 
            : base(examplesSource, settings)
        {
        }

        public override void TrainModel(NeuralNetwork network)
        {
            base.TrainModel(network);
            for ( ; Epoch <= EpochsCount; Epoch++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                double loss = 0;    
                var correct = 0;
                var examplesPassed = 0;
                foreach (var example in TrainingExamples)
                {
                    Network.Forward(example.Input);
                    CalculateLoss(example.Output);
                    loss += Loss[0];
                    Network.Backward(example.Output);
                    
                    if (CheckModelOnCorrectOutput(example.Output))
                        correct++;
                    
                    CorrectWeights();
                    
                    var result = new IterationResult
                    {
                        ExamplesPassed = examplesPassed,
                        Accuracy = correct,
                        IterationTime = sw.Elapsed,
                        Loss = loss,
                        Iteration = Iteration
                    };
                    
                    RaiseIterationFinishedEvent(result);
                    
                    sw.Restart();
                    loss = 0;
                    correct = 0;

                    Iteration++;
                }
                RaiseEpochFinishedEvent(new EpochResult());
            }
        }
    }
}