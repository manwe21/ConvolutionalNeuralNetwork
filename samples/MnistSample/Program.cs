using System;
using System.Collections.Generic;
using Network;
using Network.Model;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.NeuralMath.Cpu;
using Network.NeuralMath.Functions.LossFunctions;
using Network.NeuralMath.Gpu;
using Network.Serialization.Serializers;
using Training.Data;
using Training.Metrics;
using Training.Optimizers.Cpu;
using Training.Optimizers.Gpu;
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
            Global.ComputationType = ComputationType.Gpu;
            NeuralLayeredNetwork network = new NeuralLayeredNetwork(new Shape(BatchSize, 1, 28, 28));
            network
                .Conv(64, 3, 1, new HeInitializer())
                .Relu()
                .MaxPool(2, 2)
                .Conv(64, 3, 1, new HeInitializer())
                .Relu()
                .MaxPool(2, 2)
                .Flatten()
                .Fully(128, new HeInitializer())
                .Relu()
                .Fully(10, new HeInitializer())
                .Softmax();
            
            _trainExamples = Dataset.CreateTrainDataset(BatchSize);
            _testExamples = Dataset.CreateTestDataset(BatchSize);
            BaseTrainer trainer = new MiniBatchTrainer(_trainExamples, new MiniBatchTrainerSettings
            {
                EpochsCount = 1,
                BatchSize = BatchSize,
                LossFunction = new CrossEntropy(),
                Optimizer = new GpuAdam(0.001f),
                Metric = new ClassificationAccuracy()
            });
            trainer.AddEventHandler(new ConsoleLogger());
            trainer.TrainModel(network);
            Console.Write(network.Forward(_testExamples[0].Input));
            
            ITester tester = new ClassificationTester(_testExamples);
            //Console.WriteLine("Testing...");
            //var testResult = tester.TestModel(network);
            //Console.WriteLine(testResult);

        }
    }
}
