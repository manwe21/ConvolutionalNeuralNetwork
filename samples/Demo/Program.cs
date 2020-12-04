using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading;
using ManagedCuda;
using Network;
using Network.Model;
using Network.Model.Layers;
using Network.Model.WeightsInitializers;
using Network.NeuralMath;
using Network.NeuralMath.Cpu;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.NeuralMath.Functions.LossFunctions;
using Network.NeuralMath.Gpu;
using Network.Serialization;
using Training.Data;
using Training.Optimizers.Cpu;
using Training.Optimizers.Gpu;
using Training.Testers;
using Training.Trainers;
using Training.Trainers.EventHandlers;
using Training.Trainers.Settings;

namespace Demo
{
    delegate void DrawingCallback(int a);    
    
    static class Program    
    {
        static void Main(string[] args)
        {
            Global.ComputationType = ComputationType.Gpu;
            var dset = new Dataset<GpuTensor>(@"D:\file.dset", f => f / 255.0f, 5000);
            var network = new NeuralNetwork(new Shape(1, 3, 150, 150));
            network
                .Conv(32, 3, 1, new HeInitializer())
                .Relu()
                .MaxPool(2, 2)
                .Conv(32, 3, 1, new HeInitializer())
                .Relu()
                .MaxPool(2, 2)
                .Flatten()
                .Fully(64, new HeInitializer())
                .Relu()
                .Fully(64, new HeInitializer())
                .Relu()
                .Fully(6, new HeInitializer())
                .Softmax();
            
            BaseTrainer trainer = new MiniBatchTrainer(dset, new MiniBatchTrainerSettings
            {
                BatchSize = 32,
                EpochsCount = 20,
                LossFunction = new CrossEntropy(),
                Optimizer = new GpuAdam(1e-3f)
            });
            network.Save(@"D:\inteintel.cnn");
            //trainer.AddEventHandler(new ConsoleLogger());
            trainer.TrainModel(network);
            //network.Save(@"D:\cd2.cnn");

            /*var t = new CpuTensor(new CpuStorage(new Shape(2, 2, 4, 4)));
            var dy = new CpuTensor(new CpuStorage(new Shape(2, 2, 2, 2)));
            for (int i = 0; i < t.Size; i++)
            {
                t[i] = i;
            }
            
            for (int i = 0; i < dy.Size; i++)
            {
                dy[i] = i;
            }
            
            var res = new CpuTensor();
            
            //t.AveragePool(2, 2, res);
            t.AveragePoolDx(dy, 2, 1, res);
            //Console.WriteLine(t);
            Console.WriteLine(dy);
            Console.WriteLine(res);*/

            /*var t = new CpuTensor(new CpuStorage(new Shape(64, 64, 150, 150)));
            var res = new CpuTensor();
            var ind = new CpuTensor();
            
            Stopwatch sw = Stopwatch.StartNew();
            t.MaxPool(2, 2, res, ind);
            t.AveragePool(2, 2, res);
            
            sw.Restart();
            t.MaxPool(2, 2, res, ind);
            Console.WriteLine(sw.Elapsed);
            sw.Restart();
            t.AveragePool(2, 2, res);
            Console.WriteLine(sw.Elapsed);

            Console.WriteLine(Math.Exp(50));*/

            /*CpuTensor i = new CpuTensor(new CpuStorage(new Shape(1, 1, 1, 3)));
            CpuTensor i2 = new CpuTensor(new CpuStorage(new Shape(1, 1, 1, 3)));
            CpuTensor total = new CpuTensor(new CpuStorage(new Shape(2, 1, 1, 3)));
            CpuTensor w = new CpuTensor(new CpuStorage(new Shape(1, 1, 3, 2)));

            i[0] = 1;
            i[1] = 2;
            i[2] = 3;
            
            i2[0] = 4;
            i2[1] = 5;
            i2[2] = 6;

            total[0] = 1;
            total[1] = 2;
            total[2] = 3;
            total[3] = 4;
            total[4] = 5;
            total[5] = 6;
            
            w[0] = 0.1f;
            w[1] = 0.4f;
            w[2] = 0.2f;
            w[3] = 0.5f;
            w[4] = 0.3f;
            w[5] = 0.6f;
            
            var res = new CpuTensor();
            
            total.Dot2D(w, res);
            Console.WriteLine(res);*/
        }

    }
}
