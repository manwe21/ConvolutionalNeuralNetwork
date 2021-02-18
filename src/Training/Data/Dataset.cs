using System;
using System.Collections.Generic;
using System.IO;
using Network.NeuralMath;

 namespace Training.Data
{
    public class Dataset<TTensor> : IExamplesSource, IDisposable where TTensor : Tensor
    {
        private List<Example> _loadedExamples = new List<Example>();
        private int _realCount;

        private Func<float, float> _dataNormalizer;

        private FileStream _fileStream;
        private BinaryReader _binaryReader;

        private Shape _shape;
        private int _classes;
        
        public int LoadingBatch { get; private set; }
        public int ExamplesCount { get; private set; }
        public int TotalBatches { get; private set; }

        public Dataset(string datasetPath, int batchSize = -1)
        {
            Init(datasetPath, f => f, batchSize);
        }

        public Dataset(string datasetPath, Func<float, float> dataNormalizer, int batchSize = -1)
        {
            Init(datasetPath, dataNormalizer, batchSize);
        }

        private void Init(string datasetPath, Func<float, float> dataNormalizer, int batchSize)
        {
            FileInfo file = new FileInfo(datasetPath);
            if(!file.Exists)
                throw new ArgumentException($"File {datasetPath} is not exist");
            if(file.Extension != ".dset")
                throw new ArgumentException($"File {datasetPath} has wrong format");
            
            _fileStream = new FileStream(datasetPath, FileMode.Open);
            _binaryReader = new BinaryReader(_fileStream);

            var c = _binaryReader.ReadInt32();
            var h = _binaryReader.ReadInt32();
            var w = _binaryReader.ReadInt32();
            _shape = new Shape(1, c, h, w);
            
            ExamplesCount = _binaryReader.ReadInt32();
            _classes = _binaryReader.ReadInt32();
            
            LoadingBatch = batchSize != -1 ? batchSize : ExamplesCount;
            TotalBatches = (int)Math.Ceiling((float) ExamplesCount / LoadingBatch);
            _dataNormalizer = dataNormalizer;
            
            for (int i = 0; i < LoadingBatch; i++)
            {
                _loadedExamples.Add(new Example
                {
                    Input = (TTensor)TensorBuilder.OfType(typeof(TTensor)).OfShape(_shape),
                    Output = (TTensor)TensorBuilder.OfType(typeof(TTensor)).OfShape(new Shape(1, 1, 1, _classes))
                });
            }
        }

        private void LoadNextBatch()    
        {
            _realCount = LoadingBatch;
            for (int i = 0; i < LoadingBatch; i++)
            {
                if (_fileStream.Position >= _fileStream.Length)
                {
                    _realCount = i;
                    _fileStream.Position = 20;
                    return;
                }

                byte[] data = _binaryReader.ReadBytes(_shape.Size);
                int label = _binaryReader.ReadInt32();
                
                //normalize data
                float[] normData = new float[data.Length];
                for (int j = 0; j < data.Length; j++)
                {
                    normData[j] = _dataNormalizer(data[j]);
                }
                
                float[] outData = new float[_classes];
                outData[label] = 1;

                _loadedExamples[i].Input.Storage.Data = normData;
                _loadedExamples[i].Output.Storage.Data = outData;
            }
        }
    
        public IEnumerable<Example> GetExamples()
        {
            _fileStream.Position = 20;
            for (int b = 0; b < TotalBatches; b++)
            {
                LoadNextBatch();
                for (int i = 0; i < _realCount; i++)
                {
                    yield return _loadedExamples[i];
                }
            }
        }

        public void Dispose()
        {
            _fileStream?.Dispose();
            _binaryReader?.Dispose();
        }
    }
}
