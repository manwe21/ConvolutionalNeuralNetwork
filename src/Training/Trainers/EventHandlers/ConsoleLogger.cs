using System;

namespace Training.Trainers.EventHandlers
{
    public class ConsoleLogger : ITrainerEventHandler
    {
        public void OnIterationFinished(IterationResult result)
        {
            Console.SetCursorPosition(0, (result.Epoch - 1) * 6);
            Console.WriteLine($"Epoch: {result.Epoch}");
            Console.WriteLine($"Iteration: {result.Iteration} / {result.ExamplesPerEpoch}");
            Console.WriteLine($"Elapsed: {result.IterationTime}");
            Console.WriteLine($"Accuracy: {result.Accuracy:0.00}");
            Console.WriteLine($"Loss: {result.Loss:0.0000}");
        }    

        public void OnEpochFinished(EpochResult result)
        {
            
        }
    }
}
