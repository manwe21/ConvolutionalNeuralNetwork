using System;
using System.Collections.Generic;
using System.Diagnostics;
using Network.Model;
using Training.Data;
using Training.Trainers.Settings;

namespace Training.Trainers
{
    [Obsolete("Use mini-batch trainer with [BatchSize = examples count] instead")]
    public class FullBatchTrainer : BaseTrainer
    {
        public FullBatchTrainer(IExamplesSource trainingExamples, TrainerSettings settings) 
            : base(trainingExamples, settings)
        {
        }

        public override void TrainModel(NeuralNetwork network)
        {
            base.TrainModel(network);
            var examplesPassed = 0;
            for ( ; Epoch < EpochsCount; Epoch++, Iteration++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                double loss = 0;
                var correct = 0;
                var examplesPerIteration = 0;

                foreach (var example in TrainingExamples)
                {
                    Network.Forward(example.Input);
                    CalculateLoss(example.Output);
                    loss += Loss[0];
                    Network.Backward(example.Output);
                    
                    if (CheckModelOnCorrectOutput(example.Output))
                        correct++;
                    examplesPassed++;
                    examplesPerIteration++;
                }
                CorrectWeights();
                
                var result = new IterationResult
                {
                    ExamplesPassed = examplesPassed,
                    Iteration = Iteration,
                    IterationTime = sw.Elapsed,
                    Loss = loss / examplesPassed,
                    Accuracy = (double) correct / examplesPerIteration
                };
                RaiseIterationFinishedEvent(result);
                RaiseEpochFinishedEvent(new EpochResult());

                Iteration++;
            }
            
        }

    }
}
