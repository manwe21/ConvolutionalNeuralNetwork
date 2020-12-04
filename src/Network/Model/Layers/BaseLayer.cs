using System;
using System.Text.Json;
using Network.NeuralMath;
using Network.Serialization;

namespace Network.Model.Layers
{  
    public abstract class BaseLayer
    {
        public BaseLayer Next { get; set; }
        public BaseLayer Prev { get; set; }
        
        public Tensor Input { get; set; }
        public Tensor Output { get; set; }
        
        public Tensor InputGradient { get; set; }        
        public Tensor OutputGradient { get; set; }
        
        public Shape InputShape { get; protected set; }
        public Shape OutputShape { get; protected set; }

        protected TensorBuilder Builder;
        public bool IsInit { get; protected set; }
        
        protected BaseLayer()
        {
            Builder = TensorBuilder.Create(Global.ComputationType);
            Output = Builder.Empty();
            InputGradient = Builder.Empty();
        }

        protected BaseLayer(LayerInfo info)
        {
            InputShape = new Shape(info.InputShape.B, info.InputShape.C, info.InputShape.H, info.InputShape.W);
            OutputShape = new Shape(info.OutputShape.B, info.OutputShape.C, info.OutputShape.H, info.OutputShape.W);
            
            Builder = TensorBuilder.Create(Global.ComputationType);
            Output = Builder.Empty();
            InputGradient = Builder.Empty();
            IsInit = true;
        }

        public abstract Tensor Forward(Tensor tensor);
        
        public abstract Tensor Backward(Tensor tensor);
        
        public virtual void Initialize(Shape inputShape)
        {
            IsInit = true;
            InputShape = inputShape;
        }

        public virtual LayerInfo GetLayerInfo()
        {
            var layerInfo = new LayerInfo
            {
                LayerType = GetType().FullName,
                InputShape = new ShapeInfo(InputShape),
                OutputShape = new ShapeInfo(OutputShape)
            };
            return layerInfo;
        }

    }
}
