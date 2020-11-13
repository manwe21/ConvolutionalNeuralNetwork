using System;
using Network.NeuralMath;
using Network.NeuralMath.Cpu;
using Xunit;

namespace CpuTensorTests
{
    public abstract class TensorTests
    {
        protected abstract Tensor CreateTensor(Shape shape, float[] data);
        protected abstract Tensor CreateTensor();
        
        [Fact]
        public void Dot2D()
        {
            Shape shapeA = new Shape(1, 1, 3, 2);
            Shape shapeB = new Shape(1, 1, 2, 4);
            float[] dataA = { 0.2f, 1, 2.8f, 3, 4.1f, 5 };
            float[] dataB = { 0.1f, 2, 3.5f, 6, 4, 8.8f, 0, 3 };

            float[] res = { 4.02f, 9.2f, 0.7f, 4.2f, 12.28f, 32, 9.8f, 25.8f, 20.41f, 52.2f, 14.35f, 39.6f };

            Tensor a = CreateTensor(shapeA, dataA);
            Tensor b = CreateTensor(shapeB, dataB);
            Tensor c = CreateTensor();

            a.Dot2D(b, c);
            
            Assert.Equal(1, c.Batch);
            Assert.Equal(1, c.Channels);
            Assert.Equal(3, c.Height);
            Assert.Equal(4, c.Width);
            Assert.True(FloatComparison.AreEqual(res, c.Storage.Array));
        }

        [Fact]
        public void DotTransA2D()
        {
            Shape shapeA = new Shape(1, 1, 2, 3);
            Shape shapeB = new Shape(1, 1, 2, 4);
            float[] dataA = { 0.2f, 1, 2.8f, 3, 4.1f, 5 };
            float[] dataB = { 0.1f, 2, 3.5f, 6, 4, 8.8f, 0, 3 };
            float[] res = { 12.02f, 26.8f, 0.7f, 10.2f, 16.5f, 38.08f, 3.5f, 18.3f, 20.28f, 49.6f, 9.8f, 31.8f };
            Tensor a = CreateTensor(shapeA, dataA);
            Tensor b = CreateTensor(shapeB, dataB);
            Tensor c = CreateTensor();
            
            a.DotTransA2D(b, c);
            
            Assert.Equal(1, c.Batch);
            Assert.Equal(1, c.Channels);
            Assert.Equal(3, c.Height);
            Assert.Equal(4, c.Width);
            Assert.True(FloatComparison.AreEqual(res, c.Storage.Array));
        }

        [Fact]
        public void Transpose2D()
        {
            Shape shape = new Shape(1, 1, 2, 3);
            float[] data = { 0.2f, 1, 2.8f, 3, 4.1f, 5 };
            float[] res = { 0.2f, 3, 1, 4.1f, 2.8f, 5 };
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            
            t.Transpose2D(tRes);
            
            Assert.Equal(1, tRes.Batch);
            Assert.Equal(1, tRes.Channels);
            Assert.Equal(3, tRes.Height);
            Assert.Equal(2, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Array));
        }

        [Fact]
        public void Max()
        {
            Shape shape = new Shape(1, 1, 2, 3);
            float[] data = { 0.2f, 1, 2.8f, 3, 4.1f, 5 };
            Tensor t = CreateTensor(shape, data);
            Tensor max = CreateTensor();
            
            t.Max(max);
            
            Assert.Equal(1, max.Batch);
            Assert.Equal(1, max.Channels);
            Assert.Equal(1, max.Height);
            Assert.Equal(2, max.Width);
            Assert.Equal(5, max[0]);
            Assert.Equal(5, max[1]);
        }

        [Fact]
        public void Sum()
        {
            Shape shapeA = new Shape(1, 2, 2, 2);
            Shape shapeB = new Shape(1, 2, 2, 2);
            float[] dataA = { 0.2f, 1, 2.8f, 3, 4.1f, 5, 1, 2.2f };
            float[] dataB = { 0.1f, 2, 3.5f, 6, 4, 8.8f, 1.3f, 4.8f };
            float[] res = { 0.3f, 3, 6.3f, 9, 8.1f, 13.8f, 2.3f, 7 };
            Tensor a = CreateTensor(shapeA, dataA);
            Tensor b = CreateTensor(shapeB, dataB);

            a.Sum(b);
            
            Assert.True(FloatComparison.AreEqual(res, a.Storage.Array));
        }
        
        [Fact]
        public void Fill()
        {
            Shape shape = new Shape(1, 2, 2, 2);
            float[] data = { 0, 0, 0, 0, 0, 0, 0, 0 };
            float[] res = { 3, 3, 3, 3, 3, 3, 3, 3 };
            Tensor t = CreateTensor(shape, data);
            
            t.Fill(3);
            
            Assert.True(FloatComparison.AreEqual(res, t.Storage.Array));
        }
        
        [Fact]
        public void Img2Col()
        {    
            Shape shape = new Shape(1, 2, 3, 3);
            float[] data = { 0.2f, 1, 2.8f, 3, 4.1f, 5, 1, 2.2f, 4, 1.8f, 5, 8, 2, 2.1f, 0, 0.4f, 0.9f, 0 };
            float[] res = { 0.2f, 1, 3, 4.1f, 1, 2.8f, 4.1f, 5, 3, 4.1f, 1, 2.2f, 4.1f, 5, 2.2f, 4,
                            1.8f, 5, 2, 2.1f, 5, 8, 2.1f, 0, 2, 2.1f, 0.4f, 0.9f, 2.1f, 0, 0.9f, 0 };
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            
            t.Img2Col(2, 2, 1, tRes);
            
            Assert.Equal(1, tRes.Batch);
            Assert.Equal(1, tRes.Channels);
            Assert.Equal(8, tRes.Height);
            Assert.Equal(4, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Array));
        }
        
        [Fact]
        public void Pad()
        {
            Shape shape = new Shape(2, 2, 1, 1);
            float[] data = { 0.2f, 1, 2.8f, 3 };
            float[] res = new float[36];
            res[4] = 0.2f;
            res[13] = 1;
            res[22] = 2.8f;
            res[31] = 3;
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            
            t.Pad(1, tRes);
            
            Assert.Equal(2, tRes.Batch);
            Assert.Equal(2, tRes.Channels);
            Assert.Equal(3, tRes.Height);
            Assert.Equal(3, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Array));
        }
        
        [Fact]
        public void FullyConnectedDx()
        {
            Shape xShape = new Shape(1, 1, 1, 4);
            Shape dyShape = new Shape(1, 1, 1, 2);
            Shape wShape = new Shape(1, 1, 4, 2);
            float[] wData = { 0, 1.5f, 2, 0.2f, 5, 2.5f, 0.1f, 4 };
            float[] xData = { 2, 1.5f, 0.1f, 0.25f };
            float[] dyData = { 1, 2.5f };
            float[] dxExp = { 3.75f, 2.5f, 11.25f, 10.1f };
            Tensor x = CreateTensor(xShape, xData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dx = CreateTensor();
            
            x.FullyConnectedDx(w, dy, dx);
            
            Assert.Equal(1, dx.Batch);
            Assert.Equal(1, dx.Channels);
            Assert.Equal(1, dx.Height);
            Assert.Equal(4, dx.Width);
            Assert.True(FloatComparison.AreEqual(dxExp, dx.Storage.Array));
        }        
        
        [Fact]
        public void FullyConnectedDw()
        {
            Shape xShape = new Shape(1, 1, 1, 4);
            Shape dyShape = new Shape(1, 1, 1, 2);
            float[] xData = { 2, 1.5f, 0.1f, 0.25f };
            float[] dyData = { 1, 2.5f };
            float[] dwExp = { 2, 5, 1.5f, 3.75f, 0.1f, 0.25f, 0.25f, 0.625f };
            Tensor x = CreateTensor(xShape, xData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dw = CreateTensor();
            
            x.FullyConnectedDw(dy, dw);
            
            Assert.Equal(1, dw.Batch);
            Assert.Equal(1, dw.Channels);
            Assert.Equal(4, dw.Height);
            Assert.Equal(2, dw.Width);
            Assert.True(FloatComparison.AreEqual(dwExp, dw.Storage.Array));
        }
        
        [Fact]    
        public void Convolution()
        {
            Shape xShape = new Shape(1, 2, 3, 3);
            Shape wShape = new Shape(2, 2, 2, 2);
            float[] xData = { 0.2f, 1, 2.8f, 3, 4.1f, 5, 1, 2.2f, 4, 1.8f, 5, 8, 2, 2.1f, 0, 0.4f, 0.9f, 0 };
            float[] wData = { 0.1f, 2, 3.5f, 6, 4, 8.8f, 1.3f, 4.8f, 1, 0, 5, 0, 0, 3.5f, 4, 0.5f };
            float[] res = { 101, 143.18f, 56.52f, 51.68f, 41.75f, 57.9f, 17.4f, 18.7f };
            Tensor x = CreateTensor(xShape, xData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor y = CreateTensor();
            
            x.Convolution(w, 1, y);
            
            Assert.Equal(1, y.Batch);
            Assert.Equal(2, y.Channels);
            Assert.Equal(2, y.Height);
            Assert.Equal(2, y.Width);
            
            Assert.True(FloatComparison.AreEqual(res, y.Storage.Array));
        }
        
        [Fact]
        public void ConvolutionDx()
        {
            Shape xShape = new Shape(1, 2, 3, 3);
            Shape wShape = new Shape(2, 2, 2, 2);
            Shape dyShape = new Shape(1, 2, 2, 2);
            float[] xData = { 0.2f, 11, 2.8f, 3, 4.1f, 5, 1, 2.2f, 4, 1.8f, 5, 8, 2, 2.1f, 0, 0.4f, 0.9f, 0 };
            float[] wData = { 0.1f, 2, 3.5f, 6, 4, 8.8f, 1.3f, 4.8f, 1, 0, 5, 0, 0, 3.5f, 4, 0.5f };
            float[] dyData = { 10, 1.1f, 6.5f, 1.6f, 4.7f, 7.9f, 7.4f, 8.7f };
            float[] dxExp = { 60, 50.35f, 65.1f, 41.03f, 43.35f, 33.03f, 59, 139.35f, 77.25f, 127.41f, 57.11f, 41.28f, 13, 83.1f, 11.25f, 70.53f, 8.86f, 6.4f };

            Tensor x = CreateTensor(xShape, xData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dx = CreateTensor();
            //Tensor dw = new CpuTensor();
            //Tensor img2ColX = new CpuTensor();    
            //x.Img2Col(2, 2, 1, img2ColX);
            
            x.ConvolutionDx(w, dy, dx);
            
            Assert.Equal(1, dx.Batch);
            Assert.Equal(2, dx.Channels);
            Assert.Equal(3, dx.Height);
            Assert.Equal(3, dx.Width);
            Assert.True(FloatComparison.AreEqual(dxExp, dx.Storage.Array));

            //x.ConvolutionDw(w, dy, new CpuTensor(), new CpuTensor(), img2ColX, dw);
        }
        
        [Fact]
        public void ConvolutionDw()
        {
            Shape xShape = new Shape(1, 2, 3, 3);
            Shape wShape = new Shape(2, 2, 2, 2);
            Shape dyShape = new Shape(1, 2, 2, 2);
            float[] xData = { 0.2f, 11, 2.8f, 3, 4.1f, 5, 1, 2.2f, 4, 1.8f, 5, 8, 2, 2.1f, 0, 0.4f, 0.9f, 0 };
            float[] wData = { 0.1f, 2, 3.5f, 6, 4, 8.8f, 1.3f, 4.8f, 1, 0, 5, 0, 0, 3.5f, 4, 0.5f };
            float[] dyData = { 10, 1.1f, 6.5f, 1.6f, 4.7f, 7.9f, 7.4f, 8.7f };
            float[] dwExp = { 40.16f, 147.73f, 44.53f, 67.2f, 39.86f, 72.45f, 26.35f, 26.85f, 145.71f, 147.66f, 73.03f, 109.85f, 81.03f, 102.24f, 36.78f, 16.53f };
            Tensor x = CreateTensor(xShape, xData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dw = CreateTensor();
            
            x.ConvolutionDw(w, dy, 1, dw);
            
            Assert.Equal(2, dw.Batch);
            Assert.Equal(2, dw.Channels);
            Assert.Equal(2, dw.Height);
            Assert.Equal(2, dw.Width);
            Assert.True(FloatComparison.AreEqual(dwExp, dw.Storage.Array));
        }
        
        [Fact]
        public void MaxPool()
        {
            Shape shape = new Shape(1, 2, 3, 3);
            float[] data = { 0.2f, 1, 2.8f, 3, 4.1f, 5, 1, 2.2f, 4, 1.8f, 5, 8, 2, 2.1f, 0, 0.4f, 0.9f, 0 };
            float[] res = { 4.1f, 5, 4.1f, 5, 5, 8, 2.1f, 2.1f};
            float[] maxInd = { 4, 5, 4, 5, 10, 11, 13, 13 };
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            Tensor tMaxInd = CreateTensor();
            
            t.MaxPool(2, 1, tRes, tMaxInd);
            
            Assert.Equal(1, tRes.Batch);
            Assert.Equal(2, tRes.Channels);
            Assert.Equal(2, tRes.Height);
            Assert.Equal(2, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Array));
            Assert.True(FloatComparison.AreEqual(maxInd, tMaxInd.Storage.Array));
        }
        
        [Fact]
        public void MaxPoolDx()    
        {
            Shape xShape = new Shape(1, 2, 3, 3);
            Shape dyShape = new Shape(1, 2, 2, 2);
            Shape maxShape = new Shape(1, 2, 2, 2);
            float[] x = new float[18];
            float[] dy = { 4.1f, 5, 4.1f, 5, 5, 8, 2.1f, 2.1f};
            float[] maxInd = { 4, 5, 4, 5, 10, 11, 13, 13 };
            float[] dx = new float[18];
            dx[4] = 4.1f;
            dx[5] = 5;
            dx[10] = 5;
            dx[11] = 8;
            dx[13] = 2.1f;
            Tensor xTensor = CreateTensor(xShape, x);
            Tensor dyTensor = CreateTensor(dyShape, dy);
            Tensor maxTensor = CreateTensor(maxShape, maxInd);
            Tensor dxTensor = CreateTensor();
            
            xTensor.MaxPoolDx(dyTensor, maxTensor, dxTensor);
            
            Assert.Equal(1, dxTensor.Batch);
            Assert.Equal(2, dxTensor.Channels);
            Assert.Equal(3, dxTensor.Height);
            Assert.Equal(3, dxTensor.Width);
            Assert.True(FloatComparison.AreEqual(dx, dxTensor.Storage.Array));
        }

        [Fact]
        public void Softmax()
        {
            Shape shape = new Shape(1, 1, 1, 5);
            float[] xData = { 1, 4.5f, 0, 2, 3.1f };
            float[] yExp = { 0.022f, 0.7299f, 0.0081f, 0.0599f, 0.1799f };
            Tensor x = CreateTensor(shape, xData);
            Tensor y = CreateTensor();
            
            x.Softmax(y, CreateTensor());

            Assert.True(FloatComparison.AreEqual(yExp, y.Storage.Array));
        }
        
        [Fact]
        public void SoftmaxDx()
        {
            Shape yShape = new Shape(1, 1, 1, 5);
            Shape dyShape = new Shape(1, 1, 1, 5);
            float[] yData = { 1, 4.5f, 0, 2, 3.1f };
            float[] dyData = { 0.5f, 3, 0.4f, 1, 2.5f };
            float[] dxExp = { -23.25f, -93.375f, 0, -45.5f, -65.875f };
            Tensor dx = CreateTensor();
            Tensor y = CreateTensor(yShape, yData);
            Tensor dy = CreateTensor(dyShape, dyData);
            
            y.SoftmaxDx(dy, dx);
            
            Assert.True(FloatComparison.AreEqual(dxExp, dx.Storage.Array));
        }

        [Fact]
        public void ToFlatten()
        {
            Shape shape = new Shape(2, 2, 2, 2);
            float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            Tensor x = CreateTensor(shape, data);
            Tensor y = CreateTensor();
            
            x.ToFlatten(y);
            
            Assert.Equal(1, y.Batch);
            Assert.Equal(1, y.Channels);
            Assert.Equal(1, y.Height);
            Assert.Equal(16, y.Width);
            Assert.True(FloatComparison.AreEqual(data, y.Storage.Array));
        }
            
        [Fact]
        public void FlattenDx()    
        {    
            Shape dyShape = new Shape(1, 1, 1, 16);
            Shape xShape = new Shape(2, 2, 2, 2);
            float[] dyData = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            float[] xData = new float[16];
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor x = CreateTensor(xShape, xData);
            Tensor dx = CreateTensor();
            
            x.FlattenDx(dy, dx);
            
            Assert.Equal(2, dx.Batch);
            Assert.Equal(2, dx.Channels);
            Assert.Equal(2, dx.Height);
            Assert.Equal(2, dx.Width);
            Assert.True(FloatComparison.AreEqual(dyData, dx.Storage.Array));
        }
        
    }
}
