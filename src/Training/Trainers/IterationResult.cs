using System;

namespace Training.Trainers
{
    public class IterationResult
    {
        public int Epoch { get; set; }
        public int Iteration { get; set; }
        public TimeSpan IterationTime { get; set; }
        public int ExamplesPerEpoch { get; set; }    
        public int EpochsCount { get; set; }
        public float Accuracy { get; set; }
        public float Loss { get; set; }
    }
}
