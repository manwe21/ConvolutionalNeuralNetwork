using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using Network;
using Network.NeuralMath;
using Training.Data;

namespace MnistSample    
{
    public static class Dataset        
    {
        private static string TrainImagesFile = "train-images-idx3-ubyte.gz";
        private static string TrainLabelsFile = "train-labels-idx1-ubyte.gz";
        private static string TestImagesFile = "t10k-images-idx3-ubyte.gz";
        private static string TestLabelsFile = "t10k-labels-idx1-ubyte.gz";
        
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
            Console.Clear();
        }

        public static List<Example> CreateTrainDataset(int batch)
        {
            return CreateDataset(TrainImagesFile, TrainLabelsFile, batch);
        }
            
        public static List<Example> CreateTestDataset(int batch)
        {
            return CreateDataset(TestImagesFile, TestLabelsFile, batch);
        }
        
        private static List<Example> CreateDataset(string imagesFile, string labelsFile, int batch)
        {
            var examples = new List<Example>();
            DownloadDataset();

            using var fs = new FileStream(imagesFile, FileMode.Open);
            using var fs2 = new FileStream(labelsFile, FileMode.Open);
            using var zip1 = new GZipStream(fs, CompressionMode.Decompress);
            using var zip2 = new GZipStream(fs2, CompressionMode.Decompress);
            using var reader = new BinaryReader(zip1, Encoding.UTF8);
            using var reader2 = new BinaryReader(zip2, Encoding.UTF8);
            reader.ReadInt32();    
            reader.ReadInt32();
            reader.ReadInt32();
            reader.ReadInt32();
            reader2.ReadInt32();
            reader2.ReadInt32();
            int count = 0;
            var labels = new int[batch];
            var data = new float[0];
            var outTensor = TensorBuilder.Create().OfShape(new Shape(batch, 1, 1, 10));
            var tensor = TensorBuilder.Create().OfShape(new Shape(batch, 1, 28, 28));
            while (fs.Position != fs.Length)
            {
                float[] norm = new float[784];
                byte[] b = reader.ReadBytes(784);
                for (int i = 0; i < b.Length; i++)
                {
                    norm[i] = (float) b[i] / 255;
                }

                data = data.Concat(norm).ToArray();
                int label = reader2.ReadByte();
                labels[count] = label;
                
                if (count == batch - 1)
                {
                    for (int i = 0; i < labels.Length; i++)
                    {
                        outTensor[i, 0, 0, labels[i]] = 1;
                        labels[i] = 0;
                    }

                    tensor.Storage.Data = data;
                    examples.Add(new Example
                    {
                        Input = tensor,
                        Output = outTensor
                    });
                    
                    tensor = TensorBuilder.Create().OfShape(new Shape(batch, 1, 28, 28));
                    outTensor = TensorBuilder.Create().OfShape(new Shape(batch, 1, 1, 10));
                    data = new float[0];
                    count = 0;
                    continue;
                }

                count++;
            }

            return examples;
        }
    }
}