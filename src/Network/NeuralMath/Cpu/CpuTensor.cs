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

        public override void Dot2D(Tensor b, Tensor c)
        {
            c.Storage.AllocateMemory(GetDot2DShape(Storage.Shape, b.Storage.Shape));
            
            int m = Height;
            int n = b.Width;
            int k = Width;

            int alpha = 1;
            int beta = 0;    

            int lda = k;
            int ldb = n;
            int ldc = n;

            //MKL row major dgemm
            Blas.gemm(Layout.RowMajor, Trans.No, Trans.No, m, n, k, alpha, Storage.Data, lda, b.Storage.Data, ldb, beta, c.Storage.Data, ldc);
        }

        public override void Dot2D(Tensor b, int ha, int wa, int hb, int wb, Shape resultShape, Tensor c)
        {
            c.Storage.AllocateMemory(resultShape ?? new Shape(1, 1, ha, wb));
            
            int m = ha;
            int n = wb;
            int k = wa;

            int alpha = 1;
            int beta = 0;    

            int lda = k;
            int ldb = n;
            int ldc = n;
            
            Blas.gemm(Layout.RowMajor, Trans.No, Trans.No, m, n, k, alpha, Storage.Data, lda, b.Storage.Data, ldb, beta, c.Storage.Data, ldc);
        }

        public override void Transpose2D(Tensor result)
        {
            result.Storage.AllocateMemory(new Shape(1, 1, Width, Height));

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
            result.Storage.AllocateMemory(new Shape(Batch, 1, 1, 2));

            for (int b = 0; b < Batch; b++)
            {
                var max = this[b, 0, 0, 0];
                var maxI = 0;
                for (int c = 0; c < Channels; c++)
                {
                    for (int i = 0; i < Height; i++)
                    {
                        for (int j = 1; j < Width; j++)
                        {
                            var el = this[b, c, i, j];
                            if (el > max)
                            {
                                max = el;
                                maxI = c * Width * Height + i * Width + j;
                            }
                        }
                    }
                }

                result[b, 0, 0, 0] = max;
                result[b, 0, 0, 1] = maxI;
            }
            
        }

        public override void Average(Tensor result)
        {
            result.Storage.AllocateMemory(new Shape(Batch, 1, 1, 1));

            var sizePerBatch = Size / Batch;
            Parallel.For(0, Batch, b =>
            {
                float sum = 0;
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    sum += this[i];
                }
                result[b] = sum / sizePerBatch;
            });
        }

        public override void Pad(int value, Tensor result)
        {
            result.Storage.AllocateMemory(GetPaddingShape(Storage.Shape, value));

            var endI = result.Height - value;
            var endJ = result.Width - value;
            
            Parallel.For(0, Batch, b =>
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int i = value; i < endI; i++)
                    {
                        for (int j = value; j < endJ; j++)
                        {
                            result[b, c, i, j] = this[b, c, i - value, j - value];
                        }
                    }
                }
            });
        }

        public override void PadDx(int value, Tensor dy, Tensor result)
        {
            result.Storage.AllocateMemory(new Shape(dy.Batch, dy.Channels, dy.Height - 2 * value, dy.Width - 2 * value));

            var endI = dy.Width - value;
            var endJ = dy.Height - value;
            
            Parallel.For(0, Batch, b =>
            {
                for (int c = 0; c < dy.Channels; c++)
                {
                    for (int i = value; i < endI; i++)
                    {
                        for (int j = value; j < endJ; j++)
                        {
                            result[b, c, i - value, j - value] = dy[b, c, i, j];
                        }
                    }
                }
            });

        }

        public override void Sum(Tensor tensor)
        {
            var sizePerBatch = Size / Batch;
            Parallel.For(0, Batch, b =>
            {
                for (int i = b * sizePerBatch; i < sizePerBatch + b * sizePerBatch; i++)
                {
                    this[i] += tensor[i];
                }
            });
        }

        public override void Sum(Tensor tensor, Tensor result)
        {
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());

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
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());

            int sectorSize = Height * Width;
            int sectorsCount = Batch * Channels;
            for (int i = 0; i < Size; i++)
            {
                int localI = i % sectorSize;
                int sectorI = i / sectorsCount;

                result[i] = this[sectorI * sectorSize + sectorSize - localI - 1];
            }
        }

        public override void Im2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
            result.Storage.AllocateMemory(GetImg2ColShape(Storage.Shape, kernelH, kernelW, stride));

            var convByRow = (Width - kernelW) / stride + 1;
            var convByCol = (Height - kernelH) / stride + 1;
            var khw = kernelH * kernelW;
            var convSq = convByCol * convByRow;

            Parallel.For(0, Batch, b =>
            {
                var st = convSq * b;
                var lim = convSq + convSq * b;
                for (int i = 0; i < result.Height; i++)
                {
                    int c = i / khw;
                    int kernelIndex = i % khw;
                    int kernelI = kernelIndex / kernelH;
                    int kernelJ = kernelIndex % kernelW;
                    for (int j = st; j < lim; j++)
                    {
                        int kernelStartPointI = j % convSq / convByRow * stride;
                        int kernelStartPointJ = j % convSq % convByRow * stride;
                        int h = kernelStartPointI + kernelI;
                        int w = kernelStartPointJ + kernelJ;
                        result[i, j] = this[b, c, h, w];
                    }
                }

            });
        }

        public override void Col2Im(Shape outShape, Tensor result) 
        {
            result.Storage.AllocateMemory(outShape);

            int wh = outShape[2] * outShape[3];
    
            Parallel.For(0, outShape[0], b =>
            {
                var st = b * wh;
                var lim = b * wh + wh;
                for (int i = 0; i < Height; i++)
                {
                    for (int j = st; j < lim; j++)
                    {
                        int h = j % wh / wh;
                        int w = j % wh % wh; 
                        result[b, i, h, w] = this[i, j];
                    }
                }
            });
        }

        private void Map(Func<float, float> func, Tensor result)
        {
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var sizePerBatch = Size / Batch;
            Parallel.For(0, Batch, b =>
            {
                var start = sizePerBatch * b;
                var end = sizePerBatch * b + sizePerBatch;
                for (int i = start; i < end; i++)
                {
                    result[i] = func(this[i]);
                }
            });

        }

        private void Map2(Func<float, int, float> func, Tensor result)
        {
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());

            for (int i = 0; i < Size; i++)
            {
                result[i] = func(this[i], i);
            }
        }

        public override void FullyConnectedDx(Tensor weights, Tensor dy, Tensor transBuffer, Tensor dx)
        {
            if (!dx.Storage.IsMemoryAllocated)
            {
                dx.Storage.AllocateMemory(new Shape(Batch, Channels, Height, Width));
            }
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

        public override void Convolution(Tensor filters, int stride, Tensor img2ColBuffer, Tensor dotBuffer, Tensor result)
        {
            result.Storage.AllocateMemory(GetConvolutionalShape(Storage.Shape, filters.Storage.Shape, stride, 0));

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
        
        public override void ConvolutionDx
        (
            Tensor filters,
            Tensor dy,
            Tensor paddingBuffer,
            Tensor img2ColBuffer,
            Tensor reshapedWBuffer,
            Tensor rotBuffer,
            Tensor dotBuffer,
            Tensor dx
        )
        {
            dy.Pad(Width - dy.Width, paddingBuffer);
            paddingBuffer.Im2Col(filters.Height, filters.Width, 1, img2ColBuffer);
            filters.Rotate180(rotBuffer);
            rotBuffer.To2DByRows(reshapedWBuffer);
            reshapedWBuffer.Dot2D(img2ColBuffer, dotBuffer);
            dotBuffer.ReshapeForBatches(Storage.Shape, dx);
        }

        public override void ConvolutionDw
        (
            Tensor filters,
            Tensor dy,
            Tensor dy2DBuffer,
            Tensor dotBuffer,
            Tensor img2ColX,
            Tensor dw
        )
        {
            dw.Storage.AllocateMemory(new Shape(filters.Batch, filters.Channels, filters.Height, filters.Width));
            
            dy.To2DByColumns(dy2DBuffer);
            img2ColX.Dot2D(dy2DBuffer, dotBuffer);
            
            //TODO Create Transpose method with custom tensor dimensions
            var index = 0;
            for (int j = 0; j < dotBuffer.Width; j++)
            {
                for (int i = 0; i < dotBuffer.Height; i++)
                {
                    dw[index] = dotBuffer[i, j];
                    index++;
                }
            }
            
        }    
        
        public override void MaxPool(int poolSize, int stride, Tensor result, Tensor indexes)    
        {
            result.Storage.AllocateMemory(GetPoolingShape(Storage.Shape, poolSize, stride));

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
                indexes[i] = b * Channels * wh + c * wh + Width * y + x;
            });

        }

        public override void MaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor dx)
        {
            dx.Storage.AllocateMemory(this.Storage.Shape.GetCopy());

            for (int i = 0; i < maxIndexes.Size; i++)
            {
                dx[(int)maxIndexes[i]] = dy[i];
            }
            
        }

        public override void Activation(IFunction function, Tensor result)
        {
            this.Map(function.Process, result);
        }

        public override void ActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var sizePerBatch = Size / Batch;
            Parallel.For(0, Batch, b =>
            {
                var start = sizePerBatch * b;
                var end = sizePerBatch * b + sizePerBatch;
                for (int i = start; i < end; i++)
                {
                    dx[i] = function.Derivative(this[i]) * dy[i];
                }
            });

        }

        public override void Softmax(Tensor result, Tensor maxBuffer)
        {
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());

            this.Max(maxBuffer);
            var sizePerBatch = Size / Batch;
            for (int b = 0; b < Batch; b++)
            {
                var denominator = 0.0f;
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    denominator += MathF.Exp(this[i] - maxBuffer[b * 2]);
                }
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    result[i] = MathF.Exp(this[i] - maxBuffer[b * 2]) / denominator;
                }
            }
            
        }

        public override void SoftmaxDx(Tensor dy, Tensor dx)
        {
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());

            var sizePerBatch = Size / Batch;
            
            //Last layer is usually quite small, so Parallel.For will affect performance
            for (int b = 0; b < Batch; b++)
            {
                for (int i = b * sizePerBatch; i < b * sizePerBatch + sizePerBatch; i++)
                {
                    float sum = 0.0f;
                    for (int j = b * sizePerBatch; j < b * sizePerBatch + sizePerBatch; j++)
                    {
                        float d;
                        if (i == j)
                        {
                            d = (1 - this[i]) * this[j];
                        }
                        else d = -this[i] * this[j];
                        sum += d * dy[j];
                    }
                    dx[i] = sum;
                }
            }
        }    

        public override void Loss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            loss.Storage.AllocateMemory(new Shape(Batch, 1, 1, 1));

            lossFunction.Process(this, correct, loss);
        }

        public override void LossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            dy.Storage.AllocateMemory(Storage.Shape.GetCopy());

            lossFunction.Derivative(this, correct, dy);
        }

        public override void ToFlatten(Tensor result)
        {
            result.Storage.AllocateMemory(GetFlattenShape(Storage.Shape));

            result.Storage.Data = Storage.Data;
        }

        public override void FlattenDx(Tensor dy, Tensor dx)
        {
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());

            dx.Storage.Data = dy.Storage.Data;
        }

        public override void To2DByRows(Tensor result)
        {
            result.Storage.AllocateMemory(Get2DByRowsShape(Storage.Shape));

            Parallel.For(0, Channels, c =>
            {
                var wI = 0;
                for (int b = 0; b < Batch; b++)
                {
                    for (int i = 0; i < Height; i++)
                    {
                        for (int j = 0; j < Width; j++)
                        {
                            result[c, wI] = this[b, c, i, j];
                            wI++;
                        }
                    }
                }
            });
        }

        public override void To2DByColumns(Tensor result)
        {
            result.Storage.AllocateMemory(Get2DByColumnsShape(Storage.Shape));
                
            Parallel.For(0, Channels, c =>
            {
                var count = 0;
                for (int b = 0; b < Batch; b++)
                {
                    for (int i = 0; i < Height; i++)
                    {
                        for (int j = 0; j < Width; j++)
                        {
                            result[count, c] = this[b, c, i, j];
                            count++;
                        }
                    }
                }
            });
        }

        public override void ReshapeForBatches(Shape resultShape, Tensor result)
        {
            result.Storage.AllocateMemory(resultShape);
            
            Parallel.For(0, result.Channels, c =>
            {
                for (var b = 0; b < result.Batch; b++)
                {
                    int count = 0;
                    int widthPerBatch = Width / result.Batch;
                    for (int i = 0; i < result.Height; i++)
                    {
                        for (int j = 0; j < result.Width; j++)
                        {
                            result[b, c, i, j] = this[c, b * widthPerBatch + count];
                            count++;
                        }
                    }
                }
            });
        }
    }
}
