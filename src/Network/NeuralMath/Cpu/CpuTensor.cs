using System;
using System.Threading.Tasks;
using MKLNET;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.NeuralMath.Functions.LossFunctions;

namespace Network.NeuralMath.Cpu
{
    public class CpuTensor : Tensor
    {
        public CpuTensor() : base(new CpuStorage())
        {
        }

        public CpuTensor(CpuStorage storage) : base(storage)
        {
        }
        
        public static CpuBuilder Build => new CpuBuilder();

        public override void Dot2D(Tensor tensor, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetDot2DShape(Storage.Shape, tensor.Storage.Shape));
            }

            int m = Height;
            int n = tensor.Width;
            int k = Width;

            int alpha = 1;
            int beta = 0;    

            int lda = k;
            int ldb = n;
            int ldc = n;

            //MKL row major dgemm
            Blas.sgemm(Layout.RowMajor, Transpose.No, Transpose.No, m, n, k, alpha, Storage.Array, lda, tensor.Storage.Array, ldb, beta, result.Storage.Array, ldc);
        }

        public override void DotTransA2D(Tensor tensor, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(new Shape(1, 1, Width, tensor.Width));
            }
            
            int m = Width;
            int n = tensor.Width;
            int k = Height;

            int alpha = 1;
            int beta = 0;    

            int lda = m;
            int ldb = n;
            int ldc = n;

            Blas.sgemm(Layout.RowMajor, Transpose.Yes, Transpose.No, m, n, k, alpha, Storage.Array, lda, tensor.Storage.Array, ldb, beta, result.Storage.Array, ldc);
        }

        public override void Transpose2D(Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(new Shape(1, 1, Width, Height));
            }
            
            //naive implementation
            for (int j = 0; j < Width; j++)
            {
                for (int i = 0; i < Height; i++)
                {
                    result[j, i] = this[i, j];
                }
            }
        }

        public override void Max(Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(2);
            }

            float max = this[0];
            int maxI = 0;
            for (int i = 0; i < Size; i++)
            {
                if (this[i] > max)
                {
                    max = this[i];
                    maxI = i;
                }
            }

            result[0] = max;
            result[1] = maxI;
        }

        public override void Pad(int value, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetPaddingShape(Storage.Shape, value));
            }
            
            var index = 0;
            for (int b = 0; b < Batch; b++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int i = value; i < result.Height - value; i++)
                    {
                        for (int j = value; j < result.Width - value; j++)
                        {
                            result[b, c, i, j] = this[index++];
                        }
                    }
                }
            }
        }

        public override void PadDx(int value, Tensor dy, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(new Shape(dy.Batch, dy.Channels, dy.Height - 2 * value, dy.Width - 2 * value));
            }

            var n = 0;
            for (int b = 0; b < dy.Batch; b++)
            {
                for (int c = 0; c < dy.Channels; c++)
                {
                    for (int i = value; i < dy.Width - value; i++)
                    {
                        for (int j = value; j < dy.Height - value; j++)
                        {
                            result[n] = dy[b, c, i, j];
                            n++;
                        }
                    }
                }
            }
            
        }

        public override unsafe void Sum(Tensor tensor)
        {
            fixed (float* p1 = Storage.Array)
            {
                fixed (float* p2 = tensor.Storage.Array)
                {
                    for (int i = 0; i < Size; i++)
                    {
                        *(p1 + i) += *(p2 + i);
                    }
                    
                }
            }
        }

        public override void Sum(Tensor tensor, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }

            for (int i = 0; i < Size; i++)
            {
                result[i] = this[i] + tensor[i];
            }
        }

        public override void Fill(float value)
        {
            for (int i = 0; i < Size; i++)
            {
                this[i] = value;
            }
        }

        public override void Fill(float value, Tensor result)
        {
            Map(e => value, result);
        }

        public override void Rotate180(Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            for (int b = 0; b < Batch; b++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int i = 0; i < Height; i++)
                    {
                        for (int j = 0; j < Width; j++)
                        {
                            result[b, c, i, j] = this[b, c, Height - i - 1, Width - j - 1];
                        }
                    }
                }
            }
            
        }

        public override void Img2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetImg2ColShape(Storage.Shape, kernelH, kernelW, stride));
            }
            
            var convByRow = (Width - kernelW) / stride + 1;
            var khw = kernelH * kernelW;
            
            Parallel.For(0, result.Height, i =>
            {
                for (int j = 0; j < result.Width; j++)
                {
                    int c = i / khw;
                    int kernelStartPointI = j / convByRow * stride;
                    int kernelStartPointJ = j % convByRow * stride;

                    int kernelIndex = i % khw;
                    int kernelI = kernelIndex / kernelH;
                    int kernelJ = kernelIndex % kernelW;

                    int h = kernelStartPointI + kernelI;
                    int w = kernelStartPointJ + kernelJ;
                    result[i, j] = this[c, h, w];
                }
            });
        }

        private void Map(Func<float, float> func, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            for (int i = 0; i < Size; i++)
            {
                result[i] = func(this[i]);
            }
            
        }

        // ReSharper disable once UnusedMember.Local
        private void Map2(Func<float, int, float> func, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Size);
                result.Storage.Shape = Storage.Shape.GetCopy();
            }
            
            for (int i = 0; i < Size; i++)
            {
                result[i] = func(this[i], i);
            }
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
            filters.Dot2D(img2ColBuffer, result);
            filters.Storage.Shape = wShape;
        }

        public override void ConvolutionDx
        (
            Tensor filters,
            Tensor dy,
            Tensor paddingBuffer,
            Tensor img2ColBuffer,
            Tensor reshapedWBuffer,
            Tensor dotBuffer,
            Tensor dx
        )
        {
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(new Shape(Batch, Channels, Height, Width));
            }
            
            if (!reshapedWBuffer.Storage.IsMemoryAllocated)
            {
                reshapedWBuffer.Storage.AllocateMemory(new Shape(1, 1, filters.Batch * filters.Height * filters.Width, filters.Channels));
            }
            
            dy.Pad(Width - dy.Width, paddingBuffer);
            paddingBuffer.Img2Col(filters.Height, filters.Width, 1, img2ColBuffer);

            for (int c = 0; c < filters.Channels; c++)
            {
                var wI = 0;
                for (int b = 0; b < filters.Batch; b++)
                {
                    for (int i = 0; i < filters.Height; i++)
                    {    
                        for (int j = 0; j < filters.Width; j++)
                        {
                            reshapedWBuffer[wI, c] = filters[b, c, i, j];
                            wI++;
                        }
                    }
                }
            }

            img2ColBuffer.DotTransA2D(reshapedWBuffer, dotBuffer);
            dx.Storage.SetData(dotBuffer.Storage.Array);
        }

        public override void ConvolutionDw
        (
            Tensor filters,
            Tensor dy,
            Tensor dyReshapedBuffer,
            Tensor dotWBuffer,
            Tensor img2ColX,
            Tensor dw
        )
        {
            if (!dw.Storage.IsMemoryAllocated)
            {
                dw.Storage.AllocateMemory(new Shape(filters.Batch, filters.Channels, filters.Height, filters.Width));
            }

            if (!dyReshapedBuffer.Storage.IsMemoryAllocated)
            {
                dyReshapedBuffer.Storage.AllocateMemory(new Shape(1, 1, dy.Height * dy.Width, dy.Channels));
            }
            
            for (int c = 0; c < dy.Channels; c++)
            {
                var ii = 0;
                for (int i = 0; i < dy.Height; i++)
                {    
                    for (int j = 0; j < dy.Width; j++)
                    {
                        dyReshapedBuffer[ii++, c] = dy[c, i, j];
                    }
                }
            }
            
            img2ColX.Dot2D(dyReshapedBuffer, dotWBuffer);
             
            var index = 0;
            for (int j = 0; j < dotWBuffer.Width; j++)
            {
                for (int i = 0; i < dotWBuffer.Height; i++)
                {
                    dw[index] = dotWBuffer[i, j];
                    index++;
                }
            }
        }    
        
        public override void MaxPool(int poolSize, int stride, Tensor result, Tensor indexes)    
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetPoolingShape(Storage.Shape, poolSize, stride));
            }

            if (!indexes.Storage.IsMemoryAllocated)
            {
                indexes.Storage.AllocateMemory(new Shape(1, 1, 1, result.Size));
            }

            var countH = result.Height;
            var countW = result.Width;
            var countC = countH * countW;
            var countB = result.Channels * countC;

            var wh = Height * Width;
            
            Parallel.For(0, result.Size, i =>
            {
                int b = i / countB;
                int c = i % countB / countC;

                int kernelLocalNum = i % countC;

                int startI = kernelLocalNum / countW * stride;
                int startJ = kernelLocalNum % countW * stride;

                var max = Single.MinValue;
                int y = 0;
                int x = 0;
                for (int ki = startI; ki < startI + poolSize; ki++)
                {
                    for (int kj = startJ; kj < startJ + poolSize; kj++)
                    {
                        var element = this[b, c, ki, kj];
                        if (element > max)
                        {
                            max = element;
                            y = ki;
                            x = kj;
                        }
                    }
                }

                result[i] = max;
                indexes[i] = c * wh + Width * y + x;
            });

        }

        public override void MaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(this.Storage.Shape.GetCopy());
            }

            for (int i = 0; i < maxIndexes.Size; i++)
            {
                result[(int)maxIndexes[i]] = dy[i];
            }
            
        }

        public void AveragePool(int poolSize, int stride, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetPoolingShape(Storage.Shape, poolSize, stride));
            }
            
            var countH = result.Height;
            var countW = result.Width;
            var countC = countH * countW;
            var countB = result.Channels * countC;

            var wh = Height * Width;
            
            Parallel.For(0, result.Size, i =>
            {
                int b = i / countB;
                int c = i % countB / countC;

                int kernelLocalNum = i % countC;

                int startI = kernelLocalNum / countW * stride;
                int startJ = kernelLocalNum % countW * stride;

                float avg = 0;
                for (int ki = startI; ki < startI + poolSize; ki++)
                {
                    for (int kj = startJ; kj < startJ + poolSize; kj++)
                    {
                        avg += this[b, c, ki, kj];
                    }
                }

                result[i] = avg / (poolSize * poolSize);
            });
        }

        public void AveragePoolDx(Tensor dy, int poolSize, int stride, Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(this.Storage.Shape.GetCopy());
            }

            for (int b = 0; b < Batch; b++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int i = 0; i < dy.Height; i++)
                    {
                        for (int j = 0; j < dy.Width; j++)
                        {
                            
                        }
                    }
                }
            }
            
        }

        public override void Activation(IFunction function, Tensor result)
        {
            this.Map(function.Process, result);
        }

        public override unsafe void ActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            fixed (float* ptr = Storage.Array)
            {
                fixed (float* resPtr = dx.Storage.Array)
                {
                    fixed (float* dyPtr = dy.Storage.Array)
                    {
                        for (int i = 0; i < Size; i++)
                        {
                            *(resPtr + i) = function.Derivative(*(ptr + i)) * *(dyPtr + i);
                        }
                    }
                    
                }
            }
            
        }

        public override void Softmax(Tensor result, Tensor maxBuffer)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }

            this.Max(maxBuffer);
            var denominator = 0.0f;

            for (int i = 0; i < Size; i++)
            {
                denominator += MathF.Exp(this[i] - maxBuffer[0]);
            }

            for (int i = 0; i < result.Size; i++)
            {
                result[i] = MathF.Exp(this[i] - maxBuffer[0]) / denominator;
            }
            
        }

        public override void SoftmaxDx(Tensor dy, Tensor dx)
        {
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }

            for (int i = 0; i < Size; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < dy.Size; j++)
                {
                    float d;
                    if (i == j)
                    {
                        d = this[j] * (1 - this[i]);
                    }
                    else d = -this[i] * this[j];

                    sum += d * dy[j];
                }
                dx[i] = sum;
            }
        }    

        public override void Loss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            if (!loss.Storage.IsMemoryAllocated)
            {
                loss.Storage.AllocateMemory(1);
            }
            loss[0] = lossFunction.Process(this, correct);
        }

        public override void LossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            if (!dy.Storage.IsMemoryAllocated)
            {
                dy.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            for (int i = 0; i < Size; i++)
            {
                dy[i] = lossFunction.Derivative(this[i], correct[i]);
            }
        }

        public override void ToFlatten(Tensor result)
        {
            if (!result.Storage.IsMemoryAllocated)
            {
                result.Storage.AllocateMemory(GetFlattenShape(Storage.Shape));
            }

            result.Storage.SetData(Storage.Array);
        }

        public override void FlattenDx(Tensor dy, Tensor dx)
        {
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            }
            
            dx.Storage.SetData(dy.Storage.Array);
        }

    }
}
