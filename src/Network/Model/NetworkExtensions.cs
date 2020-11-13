using Network.Model.Layers;
using Network.Model.WeightsInitializers;
using Network.NeuralMath.Functions.ActivationFunctions;

namespace Network.Model
{
    public static class NetworkExtensions
    {
        public static NeuralNetwork Conv(this NeuralNetwork network, int filtersCount, int kernelSize, int stride, IWeightsInitializer initializer)
        {
            network.AddLayer(new ConvolutionLayer(filtersCount, kernelSize, stride, initializer));
            return network;
        }
        
        public static NeuralNetwork Conv(this NeuralNetwork network, int filtersCount, int kernelSize, int stride)
        {
            network.AddLayer(new ConvolutionLayer(filtersCount, kernelSize, stride));
            return network;
        }

        public static NeuralNetwork MaxPool(this NeuralNetwork network, int poolSize, int stride)
        {
            network.AddLayer(new PoolingLayer(poolSize, stride));
            return network;
        }

        public static NeuralNetwork Flatten(this NeuralNetwork network)
        {
            network.AddLayer(new FlattenLayer());
            return network;
        }

        public static NeuralNetwork Fully(this NeuralNetwork network, int neuronsCount, IWeightsInitializer initializer)
        {
            network.AddLayer(new FullyConnectedLayer(neuronsCount, initializer));
            return network;
        }
        
        public static NeuralNetwork Fully(this NeuralNetwork network, int neuronsCount)
        {
            network.AddLayer(new FullyConnectedLayer(neuronsCount));
            return network;
        }

        public static NeuralNetwork Softmax(this NeuralNetwork network)
        {
            network.AddLayer(new Softmax());
            return network;
        }

        public static NeuralNetwork Relu(this NeuralNetwork network)
        {
            network.AddLayer(new ActivationLayer(new Relu()));
            return network;
        }

        public static NeuralNetwork Sigmoid(this NeuralNetwork network)
        {
            network.AddLayer(new ActivationLayer(new Sigmoid()));
            return network;
        }
        
        public static NeuralNetwork Tanh(this NeuralNetwork network)
        {
            network.AddLayer(new ActivationLayer(new Tanh()));
            return network;
        }

        public static NeuralNetwork Pad(this NeuralNetwork network, int paddingSize)
        {
            network.AddLayer(new ZeroPaddingLayer(paddingSize));
            return network;
        }

    }
}
