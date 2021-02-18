using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Network.NeuralMath;

namespace Training.Data
{
    public static class DatasetGenerator
    {
        public static void FromDirectories(string directoryPath, Shape shape, string outFilePath)
        {
            FileInfo file = new FileInfo(outFilePath);
            if(file.Extension != ".dset")
                throw new ArgumentException($"File {outFilePath} has wrong format");
            
            var directory = new DirectoryInfo(directoryPath);
            var classesDirs = directory.EnumerateDirectories().ToList();
            List<Tuple<byte[], int>> data = new List<Tuple<byte[], int>>();
            using (var stream = new FileStream(outFilePath, FileMode.OpenOrCreate))
            {
                using (var writer = new BinaryWriter(stream))    
                {
                    writer.Write(shape[1]);
                    writer.Write(shape[2]);
                    writer.Write(shape[3]);
                    int totalExamples = classesDirs.Select(d => d.GetFiles().Length).Sum();
                    writer.Write(totalExamples);
                    writer.Write(classesDirs.Count);
                    int label = 0;
                    foreach (var dir in classesDirs)
                    {
                        foreach (var img in dir.EnumerateFiles())
                        {
                            byte[] imgData = new byte[shape[1] * shape[2] * shape[3]];
                            Bitmap bitmap = new Bitmap(img.FullName);
                            bitmap = (Bitmap) bitmap.GetThumbnailImage(shape[2], shape[3], null,
                                IntPtr.Zero);
                            bitmap.ToArray(imgData);
                            data.Add(new Tuple<byte[], int>(imgData, label));
                        }
                        label++;
                    }

                    //Shuffle examples
                    Random rand = new Random();
                    for (int i = 0; i < data.Count; i++)
                    {
                        int i2 = rand.Next(0, data.Count - 1);
                        var tmp = data[i];
                        data[i] = data[i2];
                        data[i2] = tmp;
                    }    

                    foreach (var example in data)
                    {
                        writer.Write(example.Item1);
                        writer.Write(example.Item2);
                    }
                    
                }
            }
        }

        public static void FromCsv(string csvPath, string outFilePath)
        {
            throw new NotImplementedException();
        }
    }
}
