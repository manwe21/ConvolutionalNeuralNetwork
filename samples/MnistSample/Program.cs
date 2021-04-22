using System;
using System.Collections.Generic;
using Network;
using Network.Model;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.NeuralMath.Functions.LossFunctions;
using Training;
using Training.Data;
using Training.Testers;
using Training.Trainers;
using Training.Trainers.EventHandlers;
using Training.Trainers.Settings;

namespace MnistSample
{
    static class Program
    {
        private static List<Example> _trainExamples = new List<Example>();
        private static List<Example> _testExamples = new List<Example>();

        private const int BatchSize = 32;
        
        static void Main(string[] args)
        {
            Global.ComputationType = ComputationType.Cpu;
            NeuralLayeredNetwork network = new NeuralLayeredNetwork(new Shape(BatchSize, 1, 28, 28));
            network
                .Conv(8, 3, 1, new HeInitializer())
                .Relu()
                .MaxPool(2, 2)
                .Flatten()
                .Fully(64, new HeInitializer())
                .Relu()
                .Fully(10, new HeInitializer())
                .Softmax();
            
            _trainExamples = Dataset.CreateTrainDataset(BatchSize);
            _testExamples = Dataset.CreateTestDataset(BatchSize);

            var optimizerFactory = ComponentsFactories.OptimizerFactory;
            var metricFactory = ComponentsFactories.MetricFactory;
            
            BaseTrainer trainer = new MiniBatchTrainer(_trainExamples, new MiniBatchTrainerSettings
            {
                EpochsCount = 1,
                BatchSize = BatchSize,
                LossFunction = new CrossEntropy(),
                Optimizer = optimizerFactory.CreateAdam(0.01f),
                Metric = metricFactory.CreateClassificationAccuracy() //Warning: GPU metrics have not implemented yet
            });
            trainer.AddEventHandler(new ConsoleLogger());
            trainer.TrainModel(network);
            
            ITester tester = new ClassificationTester(_testExamples);
            Console.WriteLine("Testing...");
            var testResult = tester.TestModel(network);
            Console.WriteLine(testResult);

        }
    }
}
