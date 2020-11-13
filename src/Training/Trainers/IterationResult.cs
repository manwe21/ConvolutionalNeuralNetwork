using System;

namespace Training.Trainers
{
    public class IterationResult
    {
        public int Epoch { get; set; }
        public int Iteration { get; set; }
        public int ExamplesPassed { get; set; }
        public TimeSpan IterationTime { get; set; }
        public int ExamplesPerEpoch { get; set; }
        public double Accuracy { get; set; }
        public double Loss { get; set; }
    }
}
