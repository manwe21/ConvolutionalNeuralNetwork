using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using CsvHelper;
using Network.Model;
using Network.NeuralMath;
using Network.NeuralMath.Functions.LossFunctions;
using Training.Data;
using Training.Metrics;
using Training.Optimizers.Cpu;
using Training.Trainers;
using Training.Trainers.EventHandlers;
using Training.Trainers.Settings;
    
namespace BostonHousing
{
    class Program
    {
        private const int RowsCount = 506;
        private const int ParametersCount = 13;
        private const string DataFilePath = "Boston.csv";
        
        static void Main(string[] args)
        {
            #region Reading Data
            
            string[] fields = new string[ParametersCount];
            float[][] parameters = new float[RowsCount][];
            float[] output = new float[RowsCount];
            
            for (int i = 0; i < parameters.Length; i++)
            {
                parameters[i] = new float[ParametersCount];
            }
            
            using (StreamReader streamReader = new StreamReader(DataFilePath))
            {
                using (CsvReader csvReader = new CsvReader(streamReader, new CultureInfo("en-US")))
                {
                    csvReader.Configuration.Delimiter = ",";
                    
                    csvReader.Read();
                    for (int i = 0; i < ParametersCount; i++)
                    {
                        fields[i] = csvReader.GetField<string>(i + 1);
                    }
                    
                    var index = 0;
                    while (csvReader.Read())
                    {
                        for (int i = 0; i < ParametersCount; i++)
                        {
                            parameters[index][i] = csvReader.GetField<float>(i + 1);
                        }

                        output[index] = csvReader.GetField<float>(ParametersCount + 1);
                        index++;
                    }
                }
            }
            
            #endregion

            #region Data standartization
            
            float[] mean = new float[ParametersCount];
            float[] deviation = new float[ParametersCount];
            for (int i = 0; i < ParametersCount; i++)
            {
                for (int j = 0; j < RowsCount; j++)
                {
                    mean[i] += parameters[j][i];
                    deviation[i] += MathF.Pow(parameters[j][i] - mean[i], 2);
                }

                mean[i] /= RowsCount;
                deviation[i] = MathF.Sqrt(deviation[i]) / RowsCount;
            }

            for (int i = 0; i < RowsCount; i++)
            {
                for (int j = 0; j < ParametersCount; j++)
                {
                    parameters[i][j] = (parameters[i][j] - mean[j]) / deviation[j];
                }
            }
            
            #endregion

            #region Training
            
            var examples = new List<Example>();
            var builder = TensorBuilder.Create();
            for (int i = 0; i < RowsCount; i++)
            {
                var inTensor = builder.OfShape(new Shape(1, 1, 1, ParametersCount));
                var outTensor = builder.OfShape(Shape.ForScalar());

                inTensor.Storage.Data = parameters[i];
                outTensor[0] = output[i];
                examples.Add(new Example
                {
                    Input = inTensor,
                    Output = outTensor
                });
            }

            var network = new NeuralLayeredNetwork(new Shape(1, 1, 1, ParametersCount));
            network
                .Fully(64)
                .Relu()
                .Fully(1);
            
            var trainer = new MiniBatchTrainer(examples, new MiniBatchTrainerSettings
            {
                BatchSize = 32,
                EpochsCount = 50,
                LossFunction = new MeanSquaredError(),
                Metric = new MeanAbsoluteError(),
                Optimizer = new CpuAdam(0.001f)
            });
            
            trainer.AddEventHandler(new ConsoleLogger());
            trainer.TrainModel(network);
            
            #endregion
        }
    }
}
