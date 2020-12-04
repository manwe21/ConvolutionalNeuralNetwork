using System;
using System.Linq;
using System.Text;
using ManagedCuda.CudaBlas;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.NeuralMath.Functions.LossFunctions;

namespace Network.NeuralMath.Gpu
{
    public class GpuTensor : Tensor, IDisposable
    {
        private readonly GpuContext _context;
        
        //Reference to TensorStorage in parent class
        //Duplicating of reference is not beautiful but help to avoid extra casts
        private readonly GpuStorage _storage;
        
        public GpuTensor() : base(new GpuStorage())
        {
            _context = ((GpuStorage) Storage).Context;
            _storage = (GpuStorage)Storage;
        }
        
        public GpuTensor(GpuStorage storage) : base(storage)
        {
            _context = storage.Context;
            _storage = storage;
        }    
        
        public static TensorBuilder Build => new GpuBuilder();

        public override void Dot2D(Tensor tensor, Tensor result)
        {
            if(!(tensor.Storage is GpuStorage bStorage))
                throw new ArgumentException($"{ nameof(tensor) } has unsupported storage type");
            if(!(result.Storage is GpuStorage cStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");

            var shape = GetDot2DShape(Storage.Shape, tensor.Storage.Shape);
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetDot2DShape(Storage.Shape, tensor.Storage.Shape));
            }
            else if (result.Storage.Shape != shape)
            {
                result.Storage.Reshape(shape);
            }
            
            int rowA = Height;
            int colA = Width;
            int colB = tensor.Width;
            int colC = result.Width;

            float alpha = 1.0f;
            float beta = 0;

            var dA = _storage.DeviceStorage;
            var dB = bStorage.DeviceStorage;
            var dC = cStorage.DeviceStorage;
            
            _context.BlasContext.Gemm(
                Operation.NonTranspose,
                Operation.NonTranspose,
                colC,
                rowA,
                colA,
                alpha,
                dB,
                colB,
                dA,
                colA,
                beta,
                dC,
                colB);
        }

        public override void DotTransA2D(Tensor tensor, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetDot2DTransAShape(Storage.Shape, tensor.Storage.Shape));
            }
            
            if(!(tensor.Storage is GpuStorage bStorage))
                throw new ArgumentException($"{ nameof(tensor) } has unsupported storage type");
            if(!(result.Storage is GpuStorage cStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");

            int rowA = Height;
            int colA = Width;
            int colB = tensor.Width;
            int colC = result.Width;

            float alpha = 1.0f;
            float beta = 0;

            var dA = _storage.DeviceStorage;
            var dB = bStorage.DeviceStorage;
            var dC = cStorage.DeviceStorage;
            
            _context.BlasContext.Gemm(
                Operation.NonTranspose,
                Operation.Transpose,
                colC,
                colA,
                rowA,
                alpha,
                dB,
                colB,
                dA,
                colA,
                beta,
                dC,
                colB);
        }

        public override void Transpose2D(Tensor result)
        {
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(new Shape(1, 1, Width, Height));
            }

            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            
            _context.Methods.Transpose2D(dX, dRes, Height, Width);
        }

        public override void Max(Tensor result)
        {
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(new Shape(1, 1, 1, 2));
            }

            var dA = _storage.DeviceStorage;
            var dMax = resStorage.DeviceStorage;
            
            _context.Methods.Max(dA, dMax, Size);
        }

        public override void Sum(Tensor tensor)
        {
            if(!(tensor.Storage is GpuStorage bStorage))
                throw new ArgumentException($"{ nameof(tensor) } has unsupported storage type");
    
            var dA = _storage.DeviceStorage;
            var dB = bStorage.DeviceStorage;
                 
            _context.Methods.Sum(dA, dB, Size);
        }

        public override void Sum(Tensor tensor, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void Fill(float value)    
        {
            var dX = _storage.DeviceStorage;
            _context.Methods.Fill(dX, value, Size);
        }

        public override void Fill(float value, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void Rotate180(Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void Img2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetImg2ColShape(Storage.Shape, kernelW, kernelH, stride));
            }
    
            var dA = _storage.DeviceStorage;
            var dRes =resStorage.DeviceStorage;
            
            int convByRow = (Width - kernelW) / stride + 1;
            _context.Methods.Img2Col2(dA, dRes, Storage.Descriptor, kernelH, stride, result.Storage.Descriptor, convByRow);
        }

        public override void Pad(int value, Tensor result)
        {
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetPaddingShape(Storage.Shape, value));
            }

            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;

            var hw = Height * Width;
            var chw = Channels * hw;
            
            _context.Methods.Pad(dX, dRes, value, Size, Width, chw, hw, result.Channels, result.Height, result.Width);
        }

        public override void PadDx(int value, Tensor dy, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void FullyConnectedDx(Tensor weights, Tensor dy, Tensor result)
        {
            var shape = dy.Storage.Shape;
            dy.Storage.Reshape(new Shape(1, 1, dy.Width, 1));
            
            if(result.Storage.IsMemoryAllocated)
                result.Storage.Reshape(new Shape(1, 1, result.Width, 1));
            
            weights.Dot2D(dy, result);
            dy.Storage.Reshape(shape);
            result.Storage.Reshape(Storage.Shape);
        }

        public override void FullyConnectedDw(Tensor dy, Tensor result)
        {
            var shape = this.Storage.Shape;
            this.Storage.Shape = new Shape(1, 1, Width, 1);
            this.Dot2D(dy, result);
            this.Storage.Shape = shape;
        }

        public override void Convolution(Tensor filters, int stride, int padding, Tensor img2ColBuffer, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)    
            {
                result.Storage.AllocateMemory(GetConvolutionalShape(Storage.Shape, filters.Storage.Shape, stride, padding));
            }
            
            this.Img2Col(filters.Height, filters.Width, stride, img2ColBuffer);
            var wShape = filters.Storage.Shape;
            filters.Storage.Shape = new Shape(1, 1, filters.Batch, filters.Channels * filters.Height * filters.Width);
            
            //keep result`s shape and use result tensor in dot product to avoid storing another one buffer
            var shape = result.Storage.Shape;
            filters.Dot2D(img2ColBuffer, result);
            result.Storage.Reshape(shape);
            filters.Storage.Reshape(wShape);
        }

        public override void ConvolutionDx(
            Tensor filters,
            Tensor dy,
            Tensor paddingBuffer,
            Tensor img2ColBuffer,
            Tensor filters2DBuffer,
            Tensor dot2DBuffer,
            Tensor dx)
        {
            if(!(filters.Storage is GpuStorage wStorage))
                throw new ArgumentException($"{ nameof(filters) } has unsupported storage type");
            if(!(filters2DBuffer.Storage is GpuStorage wByChannelsStorage))
                throw new ArgumentException($"{ nameof(filters2DBuffer) } has unsupported storage type");
            if(!(dx.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(dx) } has unsupported storage type");
            
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(new Shape(Batch, Channels, Height, Width));
            }
            
            if (!filters2DBuffer.Storage.IsMemoryAllocated)
            {
                filters2DBuffer.Storage.AllocateMemory(new Shape(1, 1, filters.Batch * filters.Height * filters.Width, filters.Channels));
            }
            
            dy.Pad(Width - dy.Width, paddingBuffer);
            paddingBuffer.Img2Col(filters.Height, filters.Width, 1, img2ColBuffer);

            var dW = wStorage.DeviceStorage;
            var dW2D = wByChannelsStorage.DeviceStorage;
            
            _context.Methods.VerticalReshape2(dW, dW2D, filters.Storage.Descriptor, filters2DBuffer.Storage.Descriptor, filters2DBuffer.Size);
            img2ColBuffer.DotTransA2D(filters2DBuffer, dot2DBuffer);
            
            resStorage.SetData((dot2DBuffer.Storage as GpuStorage)?.DeviceStorage);
        }

        public override void ConvolutionDw(Tensor filters, Tensor dy, Tensor dy2DBuffer, Tensor dot2DBuffer, Tensor img2ColX,
            Tensor dw)
        {
            if(!(dy.Storage is GpuStorage dyStorage))
                throw new ArgumentException($"{ nameof(dy) } has unsupported storage type");
            if(!(dy2DBuffer.Storage is GpuStorage dyByChannelsStorage))
                throw new ArgumentException($"{ nameof(dy2DBuffer) } has unsupported storage type");
            
            if (!dw.Storage.IsMemoryAllocated)
                dw.Storage.AllocateMemory(new Shape(filters.Batch, filters.Channels, filters.Height, filters.Width));
            if (!dy2DBuffer.Storage.IsMemoryAllocated)
                dy2DBuffer.Storage.AllocateMemory(new Shape(1, 1, dy.Height * dy.Width, dy.Channels));
            
            var dDy = dyStorage.DeviceStorage;
            var dDy2D = dyByChannelsStorage.DeviceStorage;
            
            _context.Methods.VerticalReshape2(dDy, dDy2D, dy.Storage.Descriptor, dy2DBuffer.Storage.Descriptor, dy2DBuffer.Size);
            
            img2ColX.Dot2D(dy2DBuffer, dot2DBuffer);
            dot2DBuffer.Transpose2D(dw);
        }

        public override void MaxPool(int poolSize, int stride, Tensor result, Tensor indexes)
        {
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            if(!(indexes.Storage is GpuStorage indexesStorage))
                throw new ArgumentException($"{ nameof(indexes) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetPoolingShape(Storage.Shape, poolSize, stride));
            }

            if (!indexes.Storage.IsMemoryAllocated)
            {
                indexes.Storage.AllocateMemory(new Shape(1, 1, 1, result.Size));
            }

            var dX = _storage.DeviceStorage;
            var dMax = indexesStorage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            
            _context.Methods.MaxPool2(dX, dRes, dMax, poolSize, stride, Storage.Descriptor, result.Storage.Descriptor);
        }

        public override void MaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor result)
        {
            if(!(dy.Storage is GpuStorage dyStorage))
                throw new ArgumentException($"{ nameof(dy) } has unsupported storage type");
            if(!(maxIndexes.Storage is GpuStorage indexesStorage))
                throw new ArgumentException($"{ nameof(maxIndexes) } has unsupported storage type");
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var dDy = dyStorage.DeviceStorage;
            var dMax = indexesStorage.DeviceStorage;
            var dDx = resStorage.DeviceStorage;
            
            _context.Methods.MaxPoolDx(dDy, dMax, dDx, dy.Size);
        }

        public override void Activation(IFunction function, Tensor result)
        {
            if(!(result.Storage is GpuStorage resStorage))    
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }

            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;

            _context.Methods.Activation(dX, function, dRes, Size);
        }

        public override void ActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
            if(function is null)
                throw new ArgumentNullException(nameof(function));
            if(!(dy.Storage is GpuStorage dyStorage))
                throw new ArgumentException($"{ nameof(dy) } has unsupported storage type");
            if(!(dx.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(dx) } has unsupported storage type");
            
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }

            var dX = _storage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            var dDx = resStorage.DeviceStorage;
            
            _context.Methods.ActivationDx(dX, function, dDy, dDx, Size);
        }

        public override void Softmax(Tensor result, Tensor maxBuffer)
        {
            if(!(maxBuffer.Storage is GpuStorage maxStorage))
                throw new ArgumentException($"{ nameof(maxBuffer) } has unsupported storage type");
            if(!(result.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            //this.Max(maxBuffer);

            var dX = _storage.DeviceStorage;
            var dY = resStorage.DeviceStorage;
            var dMax = maxStorage.DeviceStorage;
            
            //CPU implementation (because of GPU reduction problems)
            float[] x = dX;
            float max = x.Max();
            var denominator = 0.0f;

            for (int i = 0; i < Size; i++)
            {
                denominator += MathF.Exp(x[i] - max);
            }

            for (int i = 0; i < result.Size; i++)
            {
                result[i] = MathF.Exp(x[i] - max) / denominator;
            }
            
            //_context.Methods.Softmax(dX, dMax, dY, Size);
        }

        public override void SoftmaxDx(Tensor dy, Tensor dx)
        {
            if(!(dy.Storage is GpuStorage dyStorage))
                throw new ArgumentException($"{ nameof(dy) } has unsupported storage type");
            if(!(dx.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(dx) } has unsupported storage type");
            
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }

            var dY = _storage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            
            _context.Methods.SoftmaxDx(dY, dDy, dRes, Size);
        }

        public override void Loss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            if(!(correct.Storage is GpuStorage tStorage))
                throw new ArgumentException($"{ nameof(correct) } has unsupported storage type");
            if(!(loss.Storage is GpuStorage lossStorage))
                throw new ArgumentException($"{ nameof(loss) } has unsupported storage type");
            
            if (!loss.Storage.IsMemoryAllocated)
            {
                loss.Storage.AllocateMemory(1);
            }
            
            var dO = _storage.DeviceStorage;
            var dT = tStorage.DeviceStorage;
            var dLoss = lossStorage.DeviceStorage;

            //CPU implementation (because of GPU reduction problems)
            //loss[0] = lossFunction.Process(this, correct);

            _context.Methods.Loss(dO, dT, dLoss, lossFunction, Size);
        }

        public override void LossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            if(!(correct.Storage is GpuStorage tStorage))
                throw new ArgumentException($"{ nameof(correct) } has unsupported storage type");
            if(!(dy.Storage is GpuStorage resStorage))
                throw new ArgumentException($"{ nameof(dy) } has unsupported storage type");
            
            if (!dy.Storage.IsMemoryAllocated)
            {
                dy.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            var dO = _storage.DeviceStorage;
            var dT = tStorage.DeviceStorage;
            var dDy = resStorage.DeviceStorage;

            _context.Methods.LossDerivative(dO, dT, dDy, lossFunction, Size);
        }

        public override void ToFlatten(Tensor result)
        {
            if(!(result.Storage is GpuStorage))
                throw new ArgumentException($"{ nameof(result) } has unsupported storage type");
            
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetFlattenShape(Storage.Shape));
            }    
            
            (result.Storage as GpuStorage).SetData(_storage.DeviceStorage);
        }

        public override void FlattenDx(Tensor dy, Tensor dx)
        {
            if(!(dy.Storage is GpuStorage dyStorage))
                throw new ArgumentException($"{ nameof(dy) } has unsupported storage type");
            if(!(dx.Storage is GpuStorage))
                throw new ArgumentException($"{ nameof(dx) } has unsupported storage type");
            
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            (dx.Storage as GpuStorage).SetData(dyStorage.DeviceStorage);
        }

        public void Dispose()
        {
            _storage?.Dispose();
        }
    }
}
