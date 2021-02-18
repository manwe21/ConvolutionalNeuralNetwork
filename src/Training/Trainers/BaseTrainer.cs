using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Network.Model;
using Network.NeuralMath;
using Network.NeuralMath.Functions.LossFunctions;
using Training.Data;
using Training.Metrics;
using Training.Optimizers;
using Training.Trainers.EventHandlers;
using Training.Trainers.Settings;

namespace Training.Trainers
{
    public abstract class BaseTrainer
    {
        public IEnumerable<Example> TrainingExamples { get; private set; }
        public int ExamplesCount { get; }
        
        public INetwork Network { get; private set; }
        public IOptimizer Optimizer { get; private set; }
        public ILossFunction LossFunction { get; private set; }
        public IMetric Metric { get; set; }
        
        //memory for Loss(t)/dt computation
        protected Tensor Dy;
        
        //memory for Loss(t)
        protected Tensor Loss;

        public int Iteration { get; protected set; }
        public int Epoch { get; protected set; }
        public int EpochsCount { get; set; }

        public event Action<IterationResult> OnIterationFinished;
        public event Action<EpochResult> OnEpochFinished;

        protected BaseTrainer(IExamplesSource examplesSource, TrainerSettings settings)
        {
            ExamplesCount = examplesSource.ExamplesCount;
            Init(examplesSource.GetExamples(), settings);  
        }
        
        protected BaseTrainer(IEnumerable<Example> trainingExamples, TrainerSettings settings)
        {
            var examples = trainingExamples.ToList();
            ExamplesCount = examples.Count;
            Init(examples, settings);  
        }

        protected void RaiseIterationFinishedEvent(IterationResult result)
        {
            OnIterationFinished?.Invoke(result);
        }

        protected void RaiseEpochFinishedEvent(EpochResult result)
        {
            OnEpochFinished?.Invoke(result);
        }

        public void AddEventHandler(ITrainerEventHandler handler)
        {
            OnIterationFinished += handler.OnIterationFinished;
            OnEpochFinished += handler.OnEpochFinished;
        }

        public void RemoveEventHandler(ITrainerEventHandler handler)
        {
            OnIterationFinished -= handler.OnIterationFinished;
            OnEpochFinished -= handler.OnEpochFinished;
        }

        private void Init(IEnumerable<Example> trainingExamples, TrainerSettings settings)
        {
            TrainingExamples = trainingExamples;
            Optimizer = settings.Optimizer;
            EpochsCount = settings.EpochsCount;
            LossFunction = settings.LossFunction;
            Metric = settings.Metric;
        }

        protected void CorrectWeights()
        {
            Parallel.ForEach(Network.GetParameters(), storage =>
            {
                Optimizer.Correct(storage.Weights, storage.Gradients, storage.Parameters, true, Iteration);
            });
        }

        public virtual void TrainModel(NeuralLayeredNetwork network)
        {
            Network = network;
            Epoch = 1;
            InitializeTraining();
        }

        protected void CalculateLoss(Tensor correct)
        {
            Network.Output.Loss(correct, LossFunction, Loss);
        }

        private void InitializeTraining()
        {
            TensorBuilder builder = TensorBuilder.Create();
            Dy = builder.Empty();
            Loss = builder.Empty();
            
            if (!(Optimizer is IParametersProvider provider)) return;

            foreach (var storage in Network.GetParameters())
            {
                var layerParams = new Dictionary<string, Tensor>();
                foreach (var parameter in provider.GetParameters())
                {
                    var tensor = TensorBuilder.Create()
                        .Filled(storage.Weights.Storage.Shape, parameter.Value);
                    layerParams.Add(parameter.Key, tensor);
                }
                storage.Parameters = layerParams;
            }
            
        }

    }
}
