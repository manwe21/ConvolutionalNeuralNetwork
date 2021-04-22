using System;
using ManagedCuda;
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

        public void Transpose2D(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, TensorDescriptor xDesc)
        {
            _kernelManager.LaunchKernel(
                "transpose2d",
                xDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc);
        }

        public void Max(CudaDeviceVariable<float> x, CudaDeviceVariable<float> max, TensorDescriptor desc)
        {
            int gridSize = desc.Batch;
            int blockSize = desc.Channels * desc.Height * desc.Width;
            _kernelManager.LaunchKernel(    
                "findMax",
                gridSize,
                blockSize,
                blockSize * sizeof(float),
                x.DevicePointer,
                max.DevicePointer,
                desc);
        }

        public void Sum(CudaDeviceVariable<float> a, CudaDeviceVariable<float> b, TensorDescriptor desc)
        {
            _kernelManager.LaunchKernel(
                "sum",
                desc.Size,
                0,
                a.DevicePointer,
                b.DevicePointer,
                desc);
        }

        public void Fill(CudaDeviceVariable<float> x, float value, TensorDescriptor desc)
        {
            _kernelManager.LaunchKernel(
                "fill",
                desc.Size,
                0,
                x.DevicePointer,
                value,
                desc);
        }

        public void Rotate180(CudaDeviceVariable<float> x, CudaDeviceVariable<float> res, TensorDescriptor xDesc)
        {
            _kernelManager.LaunchKernel(
                "rotate180",
                xDesc.Size,
                0,
                x.DevicePointer,
                res.DevicePointer,
                xDesc);
        }

        public void Im2Col(
            CudaDeviceVariable<float> x,
            CudaDeviceVariable<float> result,
            TensorDescriptor xDesc,
            int kernelSize,
            int stride,
            TensorDescriptor resDesc,
            int convByRow)
        {
            _kernelManager.LaunchKernel(
                "im2Col",
                resDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc,
                resDesc,
                convByRow,
                kernelSize);
        }

        public void Col2Im(CudaDeviceVariable<float> x,
            CudaDeviceVariable<float> result, TensorDescriptor xDesc, TensorDescriptor resDesc)
        {
            _kernelManager.LaunchKernel("col2Im", xDesc.Size, 0, x.DevicePointer, result.DevicePointer, xDesc, resDesc);
        }

        public void To2DByColumns(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, TensorDescriptor xDesc, TensorDescriptor resDesc)
        {
            _kernelManager.LaunchKernel(
                "to2DByColumns",
                xDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc,
                resDesc);
        }

        public void To2DByRows(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, TensorDescriptor xDesc, TensorDescriptor resDesc)
        {
            _kernelManager.LaunchKernel("to2DByRows",
                xDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc,
                resDesc);   
        }
        
        public void ReshapeForBatches(CudaDeviceVariable<float> x, CudaDeviceVariable<float> result, TensorDescriptor xDesc, TensorDescriptor resDesc)
        {
            _kernelManager.LaunchKernel(
                "reshapeForBatches",
                xDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                xDesc,
                resDesc);
        }
        
        public void Pad(
            CudaDeviceVariable<float> x,
            CudaDeviceVariable<float> result,
            int padSize,
            TensorDescriptor xDesc,
            TensorDescriptor resDesc)
        {
            _kernelManager.LaunchKernel(
                "pad",
                resDesc.Size,
                0,
                x.DevicePointer,
                result.DevicePointer,
                padSize,
                xDesc.Width,
                xDesc.Channels * xDesc.Height * xDesc.Width,
                xDesc.Height * xDesc.Width,
                resDesc);    
        }

        public void MaxPool(
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
        
        public void MaxPoolDx(CudaDeviceVariable<float> dy, CudaDeviceVariable<float> maxIndexes, CudaDeviceVariable<float> dx, TensorDescriptor dyDesc)
        {
            _kernelManager.LaunchKernel("maxPoolDx", dyDesc.Size, 0, dy.DevicePointer, maxIndexes.DevicePointer, dx.DevicePointer, dyDesc);
        }

        public void Activation(CudaDeviceVariable<float> x, IFunction function, CudaDeviceVariable<float> y, TensorDescriptor desc)
        {
            /*string kernelName = function switch
            {
                Relu _ => "relu_forward",
                Sigmoid _ => "sigmoid_forward",
                Tanh _ => "tanh_forward",
                _ => throw new ArgumentException(nameof(function))
            };*/
            
            var gpuExecutable = function as IGpuFunction ?? throw new ArgumentException(nameof(function));

            _kernelManager.LaunchKernel(
                gpuExecutable.ForwardKernelName,
                desc.Size,
                0,
                x.DevicePointer,
                y.DevicePointer,
                desc);
        }        

        public void ActivationDx(CudaDeviceVariable<float> x, IFunction function, CudaDeviceVariable<float> dy, CudaDeviceVariable<float> dx, TensorDescriptor desc)
        {
            var gpuExecutable = function as IGpuFunction ?? throw new ArgumentException(nameof(function));
            _kernelManager.LaunchKernel(
                gpuExecutable.BackwardKernelName,
                desc.Size,
                0,
                x.DevicePointer,
                dy.DevicePointer,
                dx.DevicePointer,
                desc);
        }

        public void Softmax(CudaDeviceVariable<float> x, CudaDeviceVariable<float> max, CudaDeviceVariable<float> y, TensorDescriptor desc)
        {
            int sizePerBatch = desc.Channels * desc.Height * desc.Width;
            _kernelManager.LaunchKernel(
                "softmax", 
                desc.Batch,
                sizePerBatch,
                sizePerBatch * sizeof(float),
                x.DevicePointer,
                max.DevicePointer,
                y.DevicePointer,
                desc);
        }
        
        public void SoftmaxDx(CudaDeviceVariable<float> y, CudaDeviceVariable<float> dy, CudaDeviceVariable<float> dx, TensorDescriptor desc)
        {
            _kernelManager.LaunchKernel(
                "softmaxDx",
                desc.Size,
                0,
                y.DevicePointer,
                dy.DevicePointer,
                dx.DevicePointer,
                desc);
        }
        
        public void Loss(CudaDeviceVariable<float> o, CudaDeviceVariable<float> t, CudaDeviceVariable<float> loss, ILossFunction lossFunction, TensorDescriptor desc)
        {
            /*var kernelName = lossFunction switch
            {
                CrossEntropy _ => "cross_entropy",
                MeanSquaredError _ => "mean_squared_error",
                _ => throw new ArgumentException(nameof(lossFunction))
            };*/
            var gpuExecutable = lossFunction as IGpuFunction ?? throw new ArgumentException(nameof(lossFunction));

            int sizePerBatch = desc.Channels * desc.Height * desc.Width;
            _kernelManager.LaunchKernel(
                gpuExecutable.ForwardKernelName,
                desc.Batch,
                sizePerBatch,
                sizePerBatch * sizeof(float), 
                o.DevicePointer,
                t.DevicePointer,
                loss.DevicePointer,
                desc);
        }
        
        public void LossDerivative(CudaDeviceVariable<float> o, CudaDeviceVariable<float> t, CudaDeviceVariable<float> dy, ILossFunction lossFunction, TensorDescriptor desc)
        {
            /*var kernelName = lossFunction switch
            {
                CrossEntropy _ => "cross_entropy_dy",
                MeanSquaredError _ => "mean_squared_dy",
                _ => throw new ArgumentException(nameof(lossFunction))
            };*/
            
            var gpuExecutable = lossFunction as IGpuFunction ?? throw new ArgumentException(nameof(lossFunction));

            
            _kernelManager.LaunchKernel(
                gpuExecutable.BackwardKernelName,
                desc.Size,
                0,
                o.DevicePointer,
                t.DevicePointer,
                dy.DevicePointer,
                desc);
        }    
        
    }
}
