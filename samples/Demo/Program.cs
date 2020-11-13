using System.Collections.Generic;
using Network;
using Network.Model;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.NeuralMath.Functions.LossFunctions;
using Training.Data;
using Training.Optimizers.Cpu;
using Training.Testers;
using Training.Trainers;
using Training.Trainers.EventHandlers;
using Training.Trainers.Settings;

namespace Demo
{
    static class Program    
    {
        private static List<Example> _trainExamples = new List<Example>();
        private static List<Example> _testExamples = new List<Example>();
        
        static void Main(string[] args)
        {
            Global.ComputationType = ComputationType.Cpu;    //Change this in order to use Gpu
            NeuralNetwork network = new NeuralNetwork(new Shape(1, 1, 28, 28));
            IWeightsInitializer weightsInitializer = new HeInitializer();
            network
                .Conv(16, 3, 1, weightsInitializer)
                .Relu()
                .MaxPool(2, 2)
                .Conv(16, 3, 1, weightsInitializer)
                .Relu()
                .MaxPool(2, 2)
                .Flatten()
                .Fully(64, weightsInitializer)
                .Relu()
                .Fully(10, weightsInitializer)
                .Softmax();
            
            _trainExamples = Dataset.CreateTrainDataset();
            _testExamples = Dataset.CreateTestDataset();
            
            BaseTrainer trainer = new MiniBatchTrainer(_trainExamples, new MiniBatchTrainerSettings
            {
                EpochsCount = 5,
                BatchSize = 32,
                LossFunction = new CrossEntropy(),
                Optimizer = new CpuAdam(1e-3f)
            });
            trainer.AddEventHandler(new ConsoleLogger());
            ITester tester = new ClassificationTester(_testExamples);
            
            //Set release mode for speed improvement
            trainer.TrainModel(network);
            tester.TestModel(network);
        }


    }
}
