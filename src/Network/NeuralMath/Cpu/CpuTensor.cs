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

        protected override void DoDot2D(Tensor b, Tensor c)
        {
            int m = Height;
            int n = b.Width;
            int k = Width;

            int alpha = 1;
            int beta = 0;    

            int lda = k;
            int ldb = n;
            int ldc = n;

            //MKL row major DGEMM
            Blas.gemm(Layout.RowMajor, Trans.No, Trans.No, m, n, k, alpha, Storage.Data, lda, b.Storage.Data, ldb, beta, c.Storage.Data, ldc);
        }

        protected override void DoDot2D(Tensor b, int hA, int wA, int hB, int wB, Shape resultShape, Tensor c)
        {
            int m = hA;
            int n = wB;
            int k = wA;

            int alpha = 1;
            int beta = 0;    

            int lda = k;
            int ldb = n;
            int ldc = n;
            
            Blas.gemm(Layout.RowMajor, Trans.No, Trans.No, m, n, k, alpha, Storage.Data, lda, b.Storage.Data, ldb, beta, c.Storage.Data, ldc);
        }

        protected override void DoTranspose2D(Tensor result)
        {
            for (int j = 0; j < Width; j++)
            {
                for (int i = 0; i < Height; i++)
                {
                    result[j, i] = this[i, j];
                }
            }
        }

        protected override void FindMax(Tensor result)
        {
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

        protected override void FindAverage(Tensor result)
        {
            var sizePerBatch = Size / Batch;
            Parallel.For(0, Batch, b =>
            {
                float sum = 0;

                var start = b * sizePerBatch;
                var end = b * sizePerBatch + sizePerBatch;
                for (int i = start; i < end; i++)
                {
                    sum += this[i];
                }
                result[b] = sum / sizePerBatch;
            });
        }

        protected override void DoPad(int value, Tensor result)
        {
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

        protected override void DoPadDx(int value, Tensor dy, Tensor dx)
        {
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
                            dx[b, c, i - value, j - value] = dy[b, c, i, j];
                        }
                    }
                }
            });

        }

        protected override void DoSum(Tensor tensor)
        {
            var sizePerBatch = Size / Batch;
            Parallel.For(0, Batch, b =>
            {
                var start = b * sizePerBatch;
                var end = sizePerBatch + b * sizePerBatch;
                for (int i = start; i < end; i++)
                {
                    this[i] += tensor[i];
                }
            });
        }

        protected override void DoSum(Tensor tensor, Tensor result)
        {
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

        protected override void DoFilling(float value, Tensor result)
        {
            Map(e => value, result);
        }

        protected override void DoRotate180(Tensor result)
        {
            int sectorSize = Height * Width;
            int sectorsCount = Batch * Channels;
            for (int i = 0; i < Size; i++)
            {
                int localI = i % sectorSize;
                int sectorI = i / sectorsCount;

                result[i] = this[sectorI * sectorSize + sectorSize - localI - 1];
            }
        }

        protected override void DoIm2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
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

        protected override void DoCol2Im(Shape outShape, Tensor result) 
        {
            int wh = outShape[2] * outShape[3];
            Parallel.For(0, outShape[0], b =>
            {
                var start = b * wh;
                var end = b * wh + wh;
                for (int i = 0; i < Height; i++)
                {
                    for (int j = start; j < end; j++)
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
            for (int i = 0; i < Size; i++)
            {
                result[i] = func(this[i], i);
            }
        }

        protected override void DoMaxPool(int poolSize, int stride, Tensor result, Tensor indexes)    
        {
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

        protected override void DoMaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor dx)
        {
            for (int i = 0; i < maxIndexes.Size; i++)
            {
                var index = (int) maxIndexes[i];
                dx[index] = dy[i];
            }
        }

        protected override void DoActivation(IFunction function, Tensor result)
        {
            this.Map(function.Process, result);
        }

        protected override void DoActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
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

        protected override void DoSoftmax(Tensor result, Tensor maxBuffer)
        {
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

        protected override void DoSoftmaxDx(Tensor dy, Tensor dx)
        {
            var sizePerBatch = Size / Batch;
            
            //Last layer is usually quite small, so Parallel.For will affect performance
            for (int b = 0; b < Batch; b++)
            {
                var start = b * sizePerBatch;
                var end = b * sizePerBatch + sizePerBatch;
                for (int i = start; i < end; i++)
                {
                    float sum = 0.0f;
                    for (int j = start; j < end; j++)
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

        protected override void DoLoss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            lossFunction.Process(this, correct, loss);
        }

        protected override void DoLossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            lossFunction.Derivative(this, correct, dy);
        }

        protected override void DoFlattening(Tensor result)
        {
            result.Storage.Data = Storage.Data;
        }

        protected override void DoFlatteningDx(Tensor dy, Tensor dx)
        {
            dx.Storage.Data = dy.Storage.Data;
        }

        protected override void Do2DReshapingByRows(Tensor result)
        {
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

        protected override void Do2DReshapingByColumns(Tensor result)
        {
            Parallel.For(0, Channels, c =>
            {
                var hI = 0;
                for (int b = 0; b < Batch; b++)
                {
                    for (int i = 0; i < Height; i++)
                    {
                        for (int j = 0; j < Width; j++)
                        {
                            result[hI, c] = this[b, c, i, j];
                            hI++;
                        }
                    }
                }
            });
        }

        protected override void DoReshapingForBatches(Shape resultShape, Tensor result)
        {
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
