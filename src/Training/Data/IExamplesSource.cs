using System.Collections.Generic;
using Network.NeuralMath;

namespace Training.Data
{
    public interface IExamplesSource
    {
        int ExamplesCount { get; }
        IEnumerable<Example> GetExamples();

    }
}