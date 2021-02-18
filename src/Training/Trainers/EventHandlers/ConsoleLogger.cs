using System;

namespace Training.Trainers.EventHandlers
{
    public class ConsoleLogger : ITrainerEventHandler
    {
        private float _loss;
        private float _accuracy;
        private int _elapsed;
        
        private const int ProgressBarLength = 20;
        
        public void OnIterationFinished(IterationResult result)
        {
            _loss += result.Loss;
            _accuracy += result.Accuracy;
            _elapsed += result.IterationTime.Milliseconds;
            
            Console.SetCursorPosition(0, (result.Epoch - 1) * 2);
            Console.Write($"{result.Epoch}/{result.EpochsCount}");
            
            DrawProgressBar(result.Iteration, result.ExamplesPerEpoch);
            
            Console.Write(" - ");
            Console.Write($"{_elapsed / result.Iteration} ms/step");
            Console.Write(" - ");
            Console.Write($"metric: {_accuracy / result.Iteration:0.0000}");
            Console.Write(" - ");
            Console.Write($"loss: {_loss / result.Iteration:0.0000}");
        }

        public void OnEpochFinished(EpochResult result)
        {
            _loss = 0;
            _accuracy = 0;
            _elapsed = 0;
        }

        private void DrawProgressBar(int passed, int total)
        {
            float ratio = (float)passed / total;
            int loaded = (int)Math.Floor(ProgressBarLength * ratio);
            
            Console.Write(" [");
            for (int i = 0; i < ProgressBarLength; i++)
            {
                if(i < loaded)
                    Console.Write("=");
                else if(i == loaded)
                    Console.Write(">");
                else Console.Write(".");
            }
            Console.Write("]");
        }
    }
}
