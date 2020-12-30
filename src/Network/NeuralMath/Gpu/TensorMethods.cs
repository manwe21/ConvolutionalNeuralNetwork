using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.NeuralMath.Functions.LossFunctions;

namespace Network.NeuralMath.Gpu
{
    public class TensorMethods
    {
        private readonly KernelManager _kernelManager;

        public TensorMethods(GpuContext context)
        {
            _kernelManager = context.KernelManager;
        }

        public void Transpose2D(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, int height, int width)
        {
            int size = height * width;
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "transpose2d",
                gridX,
                blockX,
                0,
                x.DevicePointer,
                result.DevicePointer,
                height,
                width);
        }

        public void Max(CudaDeviceVariable<float> x, CudaDeviceVariable<float> max, int size)
        {
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "findMax",
                gridX,
                blockX,
                size * sizeof(float),
                x.DevicePointer,
                max.DevicePointer,
                size);
        }

        public void Sum(CudaDeviceVariable<float> a, CudaDeviceVariable<float> b, int size)
        {
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "sum",
                gridX,
                blockX,
                0,
                a.DevicePointer,
                b.DevicePointer,
                size);
        }

        public void Fill(CudaDeviceVariable<float> x, float value, int size)
        {
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "fill",
                gridX,
                blockX,
                0,
                x.DevicePointer,
                value,
                size);
        }

        public void Rotate180(CudaDeviceVariable<float> x, CudaDeviceVariable<float> res, TensorDescriptor xDesc)
        {
            _kernelManager.LaunchKernel("rotate180",
                xDesc.Size,
                0,
                x.DevicePointer,
                res.DevicePointer,
                xDesc,
                xDesc.Size
                );
        }

        public void Img2Col2(
            CudaDeviceVariable<float> x,
            CudaDeviceVariable<float> result,
            TensorDescriptor xDesc,
            int kernelSize,
            int stride,
            TensorDescriptor resDesc,
            int convByRow)
        {
            int size = resDesc.Height * resDesc.Width;
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "img2Col",
                gridX,
                blockX,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc,
                resDesc,
                convByRow,
                kernelSize);
        }

        public void Img2Col(
            CudaDeviceVariable<float> x,
            CudaDeviceVariable<float> result,
            int channels,
            int height,
            int width,
            int kernelSize,
            int stride,
            int resHeight,
            int resWidth,
            int convByRow)
        {
            int size = resHeight * resWidth;
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "img2Col",
                gridX,
                blockX,
                0,
                x.DevicePointer,
                result.DevicePointer,
                channels,
                height,
                width,
                resHeight,
                resWidth,
                convByRow,
                kernelSize);
        }

        public void VerticalReshape(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, int xChannels, int xHeight, int xWidth, int resHeight, int resWidth, int size)
        {
            _kernelManager.LaunchKernel(
                "convDx_Reshape",
                size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                size,
                xChannels,
                xHeight,
                xWidth,
                resHeight,
                resWidth);
        }
        
        public void VerticalReshape2(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, TensorDescriptor xDesc, TensorDescriptor resDesc, int size)
        {
            _kernelManager.LaunchKernel(
                "vertical_reshape",
                size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                size,
                xDesc,
                resDesc);
        }

        public void HorizontalReshape(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, int xH, int xW, int resC, int resH, int resW, int size)
        {
            _kernelManager.LaunchKernel(
                "horizontal_Reshape",
                size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xH,
                xW,
                resC,
                resH,
                resW,
                size);
        }

        public void WToRow(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, int size, TensorDescriptor xDesc, TensorDescriptor resDesc)
        {
            _kernelManager.LaunchKernel("weightsToRow",
                x.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                x.Size,
                xDesc,
                resDesc);
        }
        
        public void HorizontalReshape2(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, TensorDescriptor xDesc, TensorDescriptor resDesc, int size)
        {
            _kernelManager.LaunchKernel(
                "horizontal_Reshape",
                size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc,
                resDesc,
                size);
        }
        
        public void Pad(
            CudaDeviceVariable<float> x,
            CudaDeviceVariable<float> result,
            int padSize,
            int size,
            int width,
            int chw,
            int hw,
            int resChannels,
            int resHeight,
            int resWidth)
        {
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "pad",
                gridX,
                blockX,
                0,
                x.DevicePointer,
                result.DevicePointer,
                padSize,
                size,
                width,
                chw,
                hw,
                resChannels,
                resHeight,
                resWidth);    
        }

        public void MaxPool(
            CudaDeviceVariable<float> x, 
            CudaDeviceVariable<float> result,
            CudaDeviceVariable<float> maxIndexes,
            int poolSize, 
            int stride,
            int xSize, 
            int xBatch, 
            int xChannels,
            int xHeight, 
            int xWidth,
            int resSize,
            int resHeight, 
            int resWidth)
        {
            _kernelManager.LaunchKernel(
                "maxPool",
                resSize,
                0,
                x.DevicePointer,
                result.DevicePointer, 
                maxIndexes.DevicePointer,
                poolSize, 
                stride, 
                xSize,
                xBatch,
                xChannels,
                xHeight,
                xWidth, 
                resSize,
                resHeight,
                resWidth);
        }        
        
        public void MaxPool2(
            CudaDeviceVariable<float> x, 
            CudaDeviceVariable<float> result,
            CudaDeviceVariable<float> maxIndexes,
            int poolSize,     
            int stride,
            TensorDescriptor xDesc,
            TensorDescriptor resDesc)    
        {
            _kernelManager.LaunchKernel(
                "maxPool",
                resDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer, 
                maxIndexes.DevicePointer,
                poolSize, 
                stride, 
                xDesc, 
                resDesc);
        }
        
        public void MaxPoolDx(CudaDeviceVariable<float> dy, CudaDeviceVariable<float> maxIndexes, CudaDeviceVariable<float> dx, int size)
        {
            _kernelManager.LaunchKernel("maxPoolDx", size, 0, dy.DevicePointer, maxIndexes.DevicePointer, dx.DevicePointer, size);
        }

        public void Activation(CudaDeviceVariable<float> x, IFunction function, CudaDeviceVariable<float> y, int size)
        {
            string kernelName = function switch
            {
                Relu _ => "relu_forward",
                Sigmoid _ => "sigmoid_forward",
                Tanh _ => "tanh_forward",
                _ => throw new ArgumentException(nameof(function))
            };
            
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                kernelName,
                gridX,
                blockX,
                0,
                x.DevicePointer,
                y.DevicePointer,
                size);
        }        

        public void ActivationDx(CudaDeviceVariable<float> x, IFunction function, CudaDeviceVariable<float> dy, CudaDeviceVariable<float> dx, int size)
        {
            var kernelName = function switch
            {
                Relu _ => "relu_backward",
                Sigmoid _ => "sigmoid_backward",
                Tanh _ => "tanh_backward",
                _ => throw new ArgumentException(nameof(function))    
            };
            
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                kernelName,
                gridX,
                blockX,
                0,
                x.DevicePointer,
                dy.DevicePointer,
                dx.DevicePointer,
                size);
        }

        public void Softmax(CudaDeviceVariable<float> x, CudaDeviceVariable<float> max, CudaDeviceVariable<float> y, int size)
        {
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "softmax",
                gridX,
                blockX,
                size * sizeof(float),
                x.DevicePointer,
                max.DevicePointer,
                y.DevicePointer,
                size);
        }
        
        public void SoftmaxDx(CudaDeviceVariable<float> y, CudaDeviceVariable<float> dy, CudaDeviceVariable<float> dx, int size)
        {
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                "softmaxDx",
                gridX,
                blockX,
                0,
                y.DevicePointer,
                dy.DevicePointer,
                dx.DevicePointer,
                size);
        }
        
        public void Loss(CudaDeviceVariable<float> o, CudaDeviceVariable<float> t, CudaDeviceVariable<float> loss, ILossFunction lossFunction, int size)
        {
            var kernelName = lossFunction switch
            {
                CrossEntropy _ => "cross_entropy",
                MeanSquaredError _ => "mean_squared_error",
                _ => throw new ArgumentException(nameof(lossFunction))
            };
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                kernelName,
                gridX,
                blockX, 
                size * sizeof(float), 
                o.DevicePointer,
                t.DevicePointer,
                loss.DevicePointer,
                size);
        }
        
        public void LossDerivative(CudaDeviceVariable<float> o, CudaDeviceVariable<float> t, CudaDeviceVariable<float> dy, ILossFunction lossFunction, int size)
        {
            var kernelName = lossFunction switch
            {
                CrossEntropy _ => "cross_entropy_dy",
                MeanSquaredError _ => "mean_squared_dy",
                _ => throw new ArgumentException(nameof(lossFunction))
            };
            
            _kernelManager.CalcDim(size, out var gridX, out var blockX);
            _kernelManager.LaunchKernel(
                kernelName,
                gridX,
                blockX,
                0,
                o.DevicePointer,
                t.DevicePointer,
                dy.DevicePointer,
                size);
        }    
        
    }
}
