using System;
using System.Collections.Generic;
using System.Linq;
using Network;
using Network.Model;
using Network.NeuralMath;
using Network.NeuralMath.Functions.LossFunctions;
using Training.Data;
using Training.Optimizers;
using Training.Trainers.EventHandlers;
using Training.Trainers.Settings;

namespace Training.Trainers
{
    public class BaseTrainer
    {
        public IEnumerable<Example> TrainingExamples { get; private set; }
        public int ExamplesCount { get; }
        
        public INetwork Network { get; private set; }
        public IOptimizer Optimizer { get; private set; }
        public ILossFunction LossFunction { get; private set; }
        
        //memory for Loss(t)/dt computation
        protected Tensor Dy;
        
        //memory for Loss(t)
        protected Tensor Loss;
        
        //memory for accuracy calc
        private Tensor _max1;
        private Tensor _max2;
        
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
        }

        protected void CorrectWeights()
        {
            foreach (var weightedLayer in Network.ParameterizedLayers)
            {
                Optimizer.Correct(weightedLayer.Weights, weightedLayer.WeightsGradient, weightedLayer.Parameters, true, Iteration);
            }
        }

        public virtual void TrainModel(NeuralNetwork network)
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
            TensorBuilder builder = TensorBuilder.Create(Global.ComputationType);
            Dy = builder.Empty();
            Loss = builder.OfShape(new Shape(1, 1, 1, 1));
            _max1 = builder.OfShape(new Shape(1, 1, 1, 2));
            _max2 = builder.OfShape(new Shape(1, 1, 1, 2));
            
            if (!(Optimizer is IParametersProvider provider)) return;
            
            foreach (var layer in Network.ParameterizedLayers)
            {
                var layerParams = new Dictionary<string, Tensor>();
                foreach (var parameter in provider.GetParameters())
                {
                    var tensor = TensorBuilder.OfType(Network.Output.GetType())
                        .Filled(layer.Weights.Storage.Shape, parameter.Value);
                    layerParams.Add(parameter.Key, tensor);
                }

                layer.Parameters = layerParams;
            }
        }
        
        protected bool CheckModelOnCorrectOutput(Tensor expectedOutput)
        {
            var output = Network.Output;
            output.Max(_max1);
            expectedOutput.Max(_max2);
            return _max1[1] == _max2[1];
        }
    }
}
