using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;
using Network;
using Network.NeuralMath;
using Network.NeuralMath.Cpu;
using Training.Data;

namespace Demo    
{
    public static class Dataset        
    {
        private static string TrainImagesFile = "train-images-idx3-ubyte.gz";
        private static string TrainLabelsFile = "train-labels-idx1-ubyte.gz";
        private static string TestImagesFile = "train-images-idx3-ubyte.gz";
        private static string TestLabelsFile = "train-labels-idx1-ubyte.gz";
        
        private static void DownloadDataset()
        {
            string address = "http://yann.lecun.com/exdb/mnist/";
            WebClient client = new WebClient();
            if (!File.Exists(TrainImagesFile))
            {
                Console.WriteLine("Downloading training images...");
                client.DownloadFile(address + TrainImagesFile, TrainImagesFile);
            }

            if (!File.Exists(TrainLabelsFile))
            {
                Console.WriteLine("Downloading training labels...");
                client.DownloadFile(address + TrainLabelsFile, TrainLabelsFile);
            }
            
            if (!File.Exists(TestImagesFile))
            {
                Console.WriteLine("Downloading testing images...");
                client.DownloadFile(address + TestImagesFile, TestImagesFile);
            }

            if (!File.Exists(TestLabelsFile))
            {
                Console.WriteLine("Downloading testing labels...");
                client.DownloadFile(address + TestLabelsFile, TestLabelsFile);
            }
        }

        public static List<Example> CreateTrainDataset()
        {
            return CreateDataset(TrainImagesFile, TrainLabelsFile);
        }
        
        public static List<Example> CreateTestDataset()
        {
            return CreateDataset(TestImagesFile, TestLabelsFile);
        }
        
        private static List<Example> CreateDataset(string imagesFile, string labelsFile)
        {
            var examples = new List<Example>();
            DownloadDataset();

            using var fs = new FileStream(TrainImagesFile, FileMode.Open);
            using var fs2 = new FileStream(TrainLabelsFile, FileMode.Open);
            using var zip1 = new GZipStream(fs, CompressionMode.Decompress);
            using var zip2 = new GZipStream(fs2, CompressionMode.Decompress);
            using var reader = new BinaryReader(zip1, Encoding.UTF8);
            using var reader2 = new BinaryReader(zip2, Encoding.UTF8);
            reader.ReadInt32();    
            reader.ReadInt32();
            reader.ReadInt32();
            reader2.ReadInt32();
            reader2.ReadInt32();
            while (fs.Position != fs.Length)
            {
                float[] norm = new float[784];
                byte[] b = reader.ReadBytes(784);
                for (int i = 0; i < b.Length; i++)
                {
                    norm[i] = (float) b[i] / 255;
                }
                int label = reader2.ReadByte();

                var outTensor = TensorBuilder.Create(Global.ComputationType).OfShape(new Shape(1, 1, 1, 10));
                outTensor[label] = 1;
                var tensor = TensorBuilder.Create(Global.ComputationType).OfShape(new Shape(1, 1, 28, 28));
                tensor.Storage.SetData(norm);
                examples.Add(new Example
                {
                    Input = tensor,
                    Output = outTensor
                });
            }

            return examples;
        }
    }
}