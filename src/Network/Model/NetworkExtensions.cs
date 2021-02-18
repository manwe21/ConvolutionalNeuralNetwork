using Network.Model.Layers;
using Network.Model.WeightsInitializers;
using Network.NeuralMath.Functions.ActivationFunctions;

namespace Network.Model
{
    public static class NetworkExtensions
    {
        public static NeuralLayeredNetwork Conv(this NeuralLayeredNetwork network, int filtersCount, int kernelSize, int stride, IWeightsInitializer initializer)
        {
            network.AddLayer(new ConvolutionLayer(filtersCount, kernelSize, stride, initializer));
            return network;
        }
        
        public static NeuralLayeredNetwork Conv(this NeuralLayeredNetwork network, int filtersCount, int kernelSize, int stride)
        {
            network.AddLayer(new ConvolutionLayer(filtersCount, kernelSize, stride));
            return network;
        }

        public static NeuralLayeredNetwork MaxPool(this NeuralLayeredNetwork network, int poolSize, int stride)
        {
            network.AddLayer(new PoolingLayer(poolSize, stride));
            return network;
        }

        public static NeuralLayeredNetwork Flatten(this NeuralLayeredNetwork network)
        {
            network.AddLayer(new FlattenLayer());
            return network;
        }

        public static NeuralLayeredNetwork Fully(this NeuralLayeredNetwork network, int neuronsCount, IWeightsInitializer initializer)
        {
            network.AddLayer(new FullyConnectedLayer(neuronsCount, initializer));
            return network;
        }
        
        public static NeuralLayeredNetwork Fully(this NeuralLayeredNetwork network, int neuronsCount)
        {
            network.AddLayer(new FullyConnectedLayer(neuronsCount));
            return network;
        }

        public static NeuralLayeredNetwork Softmax(this NeuralLayeredNetwork network)
        {
            network.AddLayer(new Softmax());
            return network;
        }

        public static NeuralLayeredNetwork Relu(this NeuralLayeredNetwork network)
        {
            network.AddLayer(new ActivationLayer(new Relu()));
            return network;
        }

        public static NeuralLayeredNetwork Sigmoid(this NeuralLayeredNetwork network)
        {
            network.AddLayer(new ActivationLayer(new Sigmoid()));
            return network;
        }
        
        public static NeuralLayeredNetwork Tanh(this NeuralLayeredNetwork network)
        {
            network.AddLayer(new ActivationLayer(new Tanh()));
            return network;
        }

        public static NeuralLayeredNetwork Pad(this NeuralLayeredNetwork network, int paddingSize)
        {
            network.AddLayer(new ZeroPaddingLayer(paddingSize));
            return network;
        }

    }
}
