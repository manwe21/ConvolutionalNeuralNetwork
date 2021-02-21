using System;
using ManagedCuda.CudaBlas;
using Network.NeuralMath.Exceptions;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.NeuralMath.Functions.LossFunctions;

namespace Network.NeuralMath.Gpu
{
    public class GpuTensor : Tensor, IDisposable
    {
        private readonly GpuContext _context;
        
        //Reference to TensorStorage in parent class
        //Duplicating of reference is not beautiful but helps to avoid extra casts
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

        public override void Dot2D(Tensor b, Tensor c)
        {
            var bStorage = b.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(b));
            var cStorage = c.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(c));
            
            c.Storage.AllocateMemory(GetDot2DShape(Storage.Shape, b.Storage.Shape));
            
            int rowA = Height;
            int colA = Width;
            int colB = b.Width;
            int colC = c.Width;

            const float alpha = 1.0f;
            const float beta = 0;

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

        public override void Dot2D(Tensor b, int ha, int wa, int hb, int wb, Shape resultShape, Tensor c)
        {
            var bStorage = b.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(b));
            var cStorage = c.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(c));

            c.Storage.AllocateMemory(resultShape ?? new Shape(1, 1,ha, wb ));
            
            int rowA = ha;
            int colA = wa;
            int colB = wb;
            int colC = c.Width;

            const float alpha = 1.0f;
            const float beta = 0;

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

        public override void Transpose2D(Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(new Shape(1, 1, Width, Height));

            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            _context.Methods.Transpose2D(dX, dRes, _storage.Descriptor);
        }

        public override void Max(Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(new Shape(Batch, 1, 1, 2));
            
            var dA = _storage.DeviceStorage;
            var dMax = resStorage.DeviceStorage;
            _context.Methods.Max(dA, dMax, _storage.Descriptor);
        }

        public override void Sum(Tensor tensor)
        {
            var bStorage = tensor.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(tensor));
    
            var dA = _storage.DeviceStorage;
            var dB = bStorage.DeviceStorage;
            _context.Methods.Sum(dA, dB, _storage.Descriptor);
        }

        public override void Sum(Tensor tensor, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void Fill(float value)    
        {
            var dX = _storage.DeviceStorage;
            _context.Methods.Fill(dX, value, _storage.Descriptor);
        }

        public override void Fill(float value, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void Rotate180(Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(this.Storage.Shape);
            
            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            _context.Methods.Rotate180(dX, dRes, this.Storage.Descriptor);
        }

        public override void Im2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(GetImg2ColShape(Storage.Shape, kernelW, kernelH, stride));
            
            var dA = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            int convByRow = (Width - kernelW) / stride + 1;
            _context.Methods.Im2Col(dA, dRes, Storage.Descriptor, kernelH, stride, result.Storage.Descriptor, convByRow);
        }

        public override void Col2Im(Shape outShape, Tensor result)
        {
            result.Storage.AllocateMemory(outShape);    
            
            var dA = _storage.DeviceStorage;
            var dRes = (result.Storage as GpuStorage)?.DeviceStorage;
            _context.Methods.Col2Im(dA, dRes, Storage.Descriptor, result.Storage.Descriptor);
        }

        public override void Pad(int value, Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(GetPaddingShape(Storage.Shape, value));

            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            _context.Methods.Pad(dX, dRes, value, Storage.Descriptor, result.Storage.Descriptor);
        }

        public override void PadDx(int value, Tensor dy, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void FullyConnectedDx(Tensor weights, Tensor dy, Tensor transBuffer, Tensor dx)
        {
            dx.Storage.AllocateMemory(new Shape(Batch, Channels, Height, Width));
            
            weights.Transpose2D(transBuffer);
            dy.Dot2D(transBuffer, dy.Batch, dy.Width, transBuffer.Height, transBuffer.Width, Storage.Shape, dx);
        }

        public override void FullyConnectedDw(Tensor dy, Tensor transBuffer, Tensor dw)
        {
            var shape = Storage.Shape;
            Storage.Shape = new Shape(1, 1, Batch, Width);
            this.Transpose2D(transBuffer);
            transBuffer.Dot2D(dy, transBuffer.Height, transBuffer.Width, dy.Batch, dy.Width, dw.Storage.Shape, dw);
            Storage.Shape = shape;
        }

        public override void Convolution(Tensor filters, int stride, int padding, Tensor img2ColBuffer, Tensor dotBuffer, Tensor result)
        {
            result.Storage.AllocateMemory(GetConvolutionalShape(Storage.Shape, filters.Storage.Shape, stride, padding));

            this.Im2Col(filters.Height, filters.Width, stride, img2ColBuffer);
            filters.Dot2D(img2ColBuffer,
                filters.Batch,
                filters.Channels * filters.Height * filters.Width,
                img2ColBuffer.Height,
                img2ColBuffer.Width,
                null,
                dotBuffer);
            dotBuffer.Col2Im(result.Storage.Shape, result);
        }

        public override void ConvolutionDx(
            Tensor filters,
            Tensor dy,
            Tensor paddingBuffer,
            Tensor img2ColBuffer,
            Tensor filters2DBuffer,
            Tensor rotBuffer,
            Tensor dot2DBuffer,
            Tensor dx)
        {
            _ = filters.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(filters));
            var wByChannelsStorage = filters2DBuffer.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(filters2DBuffer));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));

            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            filters2DBuffer.Storage.AllocateMemory(Get2DByRowsShape(filters.Storage.Shape));

            dy.Pad(Width - dy.Width, paddingBuffer);
            paddingBuffer.Im2Col(filters.Height, filters.Width, 1, img2ColBuffer);
            filters.Rotate180(rotBuffer);
            
            var dW = (rotBuffer.Storage as GpuStorage)?.DeviceStorage;
            var dW2D = wByChannelsStorage.DeviceStorage;
            
            _storage.Context.Methods.To2DByRows(dW, dW2D, filters.Storage.Descriptor, filters2DBuffer.Storage.Descriptor);
            filters2DBuffer.Dot2D(img2ColBuffer, dot2DBuffer);
            _context.Methods.ReshapeForBatches((dot2DBuffer as GpuTensor)?._storage.DeviceStorage, dxStorage.DeviceStorage, dot2DBuffer.Storage.Descriptor, dxStorage.Descriptor);
        }

        public override void ConvolutionDw(Tensor filters, Tensor dy, Tensor dy2DBuffer, Tensor dot2DBuffer, Tensor img2ColX, Tensor dw)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var dy2DStorage = dy2DBuffer.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy2DBuffer));
            
            dw.Storage.AllocateMemory(filters.Storage.Shape.GetCopy());
            dy2DBuffer.Storage.AllocateMemory(Get2DByColumnsShape(dy.Storage.Shape));
            
            var dDy = dyStorage.DeviceStorage;
            var dDy2D = dy2DStorage.DeviceStorage;
            
            _context.Methods.To2DByColumns(dDy, dDy2D, dy.Storage.Descriptor, dy2DBuffer.Storage.Descriptor);
            img2ColX.Dot2D(dy2DBuffer, dot2DBuffer);
            dot2DBuffer.Transpose2D(dw);
        }

        public override void MaxPool(int poolSize, int stride, Tensor result, Tensor indexes)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));
            var indexesStorage = indexes.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(indexes));

            result.Storage.AllocateMemory(GetPoolingShape(Storage.Shape, poolSize, stride));
            indexes.Storage.AllocateMemory(Shape.ForVector(result.Size));

            var dX = _storage.DeviceStorage;
            var dMax = indexesStorage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            _context.Methods.MaxPool(dX, dRes, dMax, poolSize, stride, Storage.Descriptor, result.Storage.Descriptor);
        }

        public override void MaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var indexesStorage = maxIndexes.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(maxIndexes));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var dDy = dyStorage.DeviceStorage;
            var dMax = indexesStorage.DeviceStorage;
            var dDx = dxStorage.DeviceStorage;
            _context.Methods.MaxPoolDx(dDy, dMax, dDx, dy.Storage.Descriptor);
        }

        public override void Activation(IFunction function, Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var dX = _storage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            _context.Methods.Activation(dX, function, dRes, _storage.Descriptor);
        }

        public override void ActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
            if(function is null) throw new ArgumentNullException(nameof(function));
            
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var dX = _storage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            var dDx = dxStorage.DeviceStorage;
            _context.Methods.ActivationDx(dX, function, dDy, dDx, _storage.Descriptor);
        }
        
        public override void Softmax(Tensor result, Tensor maxBuffer)
        {
            var maxStorage = maxBuffer.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(maxBuffer));
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));
            
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            
            this.Max(maxBuffer);

            //BUG When GPU-Softmax works together with convolution, result is wrong
            #region CPU Implementation
            
            var cpu = _storage.Data;
            var res = new float[result.Size];
            var sizePerBatch = Size / Batch;
            for (int b = 0; b < Batch; b++)
            {
                var denominator = 0.0f;
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    denominator += MathF.Exp(cpu[i] - maxBuffer[b * 2]);
                }
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    res[i] = MathF.Exp(cpu[i] - maxBuffer[b * 2]) / denominator;
                }
            }

            result.Storage.Data = res;
            
            #endregion
            
            /*var dX = _storage.DeviceStorage;
            var dY = resStorage.DeviceStorage;
            var dMax = maxStorage.DeviceStorage;

            _context.Methods.Softmax(dX, dMax, dY, _storage.Descriptor);*/
        }

        public override void SoftmaxDx(Tensor dy, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var resStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));

            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            
            var dY = _storage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            var dRes = resStorage.DeviceStorage;
            _context.Methods.SoftmaxDx(dY, dDy, dRes, _storage.Descriptor);
        }

        public override void Loss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            var tStorage = correct.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(correct));
            var lossStorage = loss.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(loss));

            loss.Storage.AllocateMemory(new Shape(Batch, 1, 1, 1));

            var dO = _storage.DeviceStorage;
            var dT = tStorage.DeviceStorage;
            var dLoss = lossStorage.DeviceStorage;
            _context.Methods.Loss(dO, dT, dLoss, lossFunction, _storage.Descriptor);
        }

        public override void LossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            var tStorage = correct.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(correct));
            var resStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));

            dy.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var dO = _storage.DeviceStorage;
            var dT = tStorage.DeviceStorage;
            var dDy = resStorage.DeviceStorage;
            _context.Methods.LossDerivative(dO, dT, dDy, lossFunction, dy.Storage.Descriptor);
        }

        public override void ToFlatten(Tensor result)
        {
            _ = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            result.Storage.AllocateMemory(GetFlattenShape(Storage.Shape));
            (result.Storage as GpuStorage)?.SetDeviceData(_storage.DeviceStorage);
        }

        public override void FlattenDx(Tensor dy, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            _ = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));

            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            
            (dx.Storage as GpuStorage)?.SetDeviceData(dyStorage.DeviceStorage);
        }

        public void Dispose()
        {
            _storage?.Dispose();
        }
    }
}
