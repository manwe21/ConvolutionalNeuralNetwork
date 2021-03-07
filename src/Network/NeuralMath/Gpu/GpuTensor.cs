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

        protected override void DoDot2D(Tensor b, Tensor c)
        {
            var bStorage = b.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(b));
            var cStorage = c.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(c));
            
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

        protected override void DoDot2D(Tensor b, int hA, int wA, int hB, int wB, Shape resultShape, Tensor c)
        {
            var bStorage = b.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(b));
            var cStorage = c.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(c));
            
            int rowA = hA;
            int colA = wA;
            int colB = wB;
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

        protected override void DoTranspose2D(Tensor result)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            var dX = _storage.DeviceStorage;
            var dRes = resultStorage.DeviceStorage;
            _context.Methods.Transpose2D(dX, dRes, _storage.Descriptor);
        }

        protected override void FindMax(Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            var dA = _storage.DeviceStorage;
            var dMax = resStorage.DeviceStorage;
            
            _context.Methods.Max(dA, dMax, _storage.Descriptor);
        }

        protected override void FindAverage(Tensor result)
        {
            throw new NotImplementedException();
        }

        protected override void DoSum(Tensor tensor)
        {
            var bStorage = tensor.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(tensor));
    
            var dA = _storage.DeviceStorage;
            var dB = bStorage.DeviceStorage;
            
            _context.Methods.Sum(dA, dB, _storage.Descriptor);
        }

        protected override void DoSum(Tensor tensor, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        public override void Fill(float value)    
        {
            var dX = _storage.DeviceStorage;
            _context.Methods.Fill(dX, value, _storage.Descriptor);
        }

        protected override void DoFilling(float value, Tensor result)
        {
            throw new System.NotImplementedException();
        }

        protected override void DoRotate180(Tensor result)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            var dX = _storage.DeviceStorage;
            var dRes = resultStorage.DeviceStorage;
            
            _context.Methods.Rotate180(dX, dRes, this.Storage.Descriptor);
        }

        protected override void DoIm2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            var dA = _storage.DeviceStorage;
            var dRes = resultStorage.DeviceStorage;
            
            var convolutionsByRow = (Width - kernelW) / stride + 1;
            _context.Methods.Im2Col(dA, dRes, Storage.Descriptor, kernelH, stride, result.Storage.Descriptor, convolutionsByRow);
        }

        protected override void DoCol2Im(Shape outShape, Tensor result)
        {
            var dA = _storage.DeviceStorage;
            var dRes = (result.Storage as GpuStorage)?.DeviceStorage;
            _context.Methods.Col2Im(dA, dRes, Storage.Descriptor, result.Storage.Descriptor);
        }

        protected override void DoPad(int value, Tensor result)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            var dX = _storage.DeviceStorage;
            var dRes = resultStorage.DeviceStorage;
            
            _context.Methods.Pad(dX, dRes, value, Storage.Descriptor, result.Storage.Descriptor);
        }

        protected override void DoPadDx(int value, Tensor dy, Tensor dx)
        {
            throw new System.NotImplementedException();
        }

        protected override void DoMaxPool(int poolSize, int stride, Tensor result, Tensor indexes)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));
            var indexesStorage = indexes.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(indexes));

            var dX = _storage.DeviceStorage;
            var dMax = indexesStorage.DeviceStorage;
            var dRes = resultStorage.DeviceStorage;
            
            _context.Methods.MaxPool(dX, dRes, dMax, poolSize, stride, Storage.Descriptor, result.Storage.Descriptor);
        }

        protected override void DoMaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var indexesStorage = maxIndexes.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(maxIndexes));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));
            
            var dDy = dyStorage.DeviceStorage;
            var dMax = indexesStorage.DeviceStorage;
            var dDx = dxStorage.DeviceStorage;
            
            _context.Methods.MaxPoolDx(dDy, dMax, dDx, dy.Storage.Descriptor);
        }

        protected override void DoActivation(IFunction function, Tensor result)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            var dX = _storage.DeviceStorage;
            var dRes = resultStorage.DeviceStorage;
            
            _context.Methods.Activation(dX, function, dRes, _storage.Descriptor);
        }

        protected override void DoActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));
            
            var dX = _storage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            var dDx = dxStorage.DeviceStorage;
            
            _context.Methods.ActivationDx(dX, function, dDy, dDx, _storage.Descriptor);
        }
        
        protected override void DoSoftmax(Tensor result, Tensor maxBuffer)
        {
            var maxStorage = maxBuffer.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(maxBuffer));
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

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

        protected override void DoSoftmaxDx(Tensor dy, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));

            var dY = _storage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            var dRes = dxStorage.DeviceStorage;
            
            _context.Methods.SoftmaxDx(dY, dDy, dRes, _storage.Descriptor);
        }

        protected override void DoLoss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            var tStorage = correct.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(correct));
            var lossStorage = loss.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(loss));

            var dO = _storage.DeviceStorage;
            var dT = tStorage.DeviceStorage;
            var dLoss = lossStorage.DeviceStorage;
            
            _context.Methods.Loss(dO, dT, dLoss, lossFunction, _storage.Descriptor);
        }

        protected override void DoLossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            var tStorage = correct.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(correct));
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));

            var dO = _storage.DeviceStorage;
            var dT = tStorage.DeviceStorage;
            var dDy = dyStorage.DeviceStorage;
            
            _context.Methods.LossDerivative(dO, dT, dDy, lossFunction, dy.Storage.Descriptor);
        }

        protected override void DoFlattening(Tensor result)
        {
            var resultStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));

            resultStorage.SetDeviceData(_storage.DeviceStorage);
        }

        protected override void DoFlatteningDx(Tensor dy, Tensor dx)
        {
            var dyStorage = dy.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dy));
            var dxStorage = dx.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(dx));

            dxStorage.SetDeviceData(dyStorage.DeviceStorage);
        }

        protected override void Do2DReshapingByRows(Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));
            
            _context.Methods.To2DByRows(_storage.DeviceStorage, resStorage.DeviceStorage, Storage.Descriptor, resStorage.Descriptor);
        }

        protected override void Do2DReshapingByColumns(Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));
            
            _context.Methods.To2DByColumns(_storage.DeviceStorage, resStorage.DeviceStorage, Storage.Descriptor, resStorage.Descriptor);
        }

        protected override void DoReshapingForBatches(Shape resultShape, Tensor result)
        {
            var resStorage = result.Storage as GpuStorage ?? throw new UnsupportedStorageException(nameof(result));
            
            _context.Methods.ReshapeForBatches(_storage.DeviceStorage, resStorage.DeviceStorage, _storage.Descriptor, result.Storage.Descriptor);
        }

        public void Dispose()
        {
            _storage.Dispose();
        }
    }
}
