using System;
using System.Collections.Generic;
using Network;
using Network.Model;
using Network.NeuralMath;
using Training.Data;

namespace Training.Testers
{
    public class ClassificationTester : ITester
    {
        private readonly IEnumerable<Example> _testExamples;

        public ClassificationTester(IExamplesSource examplesSource)
        {
            _testExamples = examplesSource.GetExamples();
        }

        public ClassificationTester(IEnumerable<Example> testExamples)
        {
            _testExamples = testExamples;
        }

        public TestResult TestModel(NeuralNetwork network)    
        {
            var result = new TestResult();
            var max1 = TensorBuilder.Create(Global.ComputationType).Empty();
            var max2 = TensorBuilder.Create(Global.ComputationType).Empty();
            foreach (var example in _testExamples)
            {
                network.Forward(example.Input);
                network.Output.Max(max1);
                example.Output.Max(max2);
                if (max1[1] == max2[1])
                    result.Successful++;
                else result.Failed++;
                result.TotalTests++;
                result.SuccessfulRatio = (float)result.Successful/result.TotalTests;
                Console.WriteLine(result);
            }
            return result;
        }
    }
}
