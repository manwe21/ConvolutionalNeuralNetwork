using Network.NeuralMath;
using Network.NeuralMath.Functions.LossFunctions;
using Xunit;

namespace TensorTests
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
            
            float[] dataA =
            {
                0.2f, 1,
                2.8f, 3,
                4.1f, 5
            };
            
            float[] dataB =
            {
                0.1f, 2, 3.5f, 6,
                4, 8.8f, 0, 3
            };

            float[] res =
            {
                4.02f, 9.2f, 0.7f, 4.2f,
                12.28f, 32, 9.8f, 25.8f,
                20.41f, 52.2f, 14.35f, 39.6f
            };

            Tensor a = CreateTensor(shapeA, dataA);
            Tensor b = CreateTensor(shapeB, dataB);
            Tensor c = CreateTensor();

            a.Dot2D(b, c);
            
            Assert.Equal(1, c.Batch);
            Assert.Equal(1, c.Channels);
            Assert.Equal(3, c.Height);
            Assert.Equal(4, c.Width);
            Assert.True(FloatComparison.AreEqual(res, c.Storage.Data));
        }

        [Fact]
        public void Dot2D_ExplicitDimensions()
        {
            Shape shapeA = new Shape(3, 1, 1, 2);
            Shape shapeB = new Shape(1, 1, 2, 4);
            Shape shapeC = new Shape(3, 1, 1, 4);
            
            float[] dataA = 
            {
                0.2f, 1,
                
                2.8f, 3,
                
                4.1f, 5
            };
            
            float[] dataB =
            {
                0.1f, 2, 3.5f, 6,
                4, 8.8f, 0, 3
            };

            float[] res =
            {
                4.02f, 9.2f, 0.7f, 4.2f,
                
                12.28f, 32, 9.8f, 25.8f,
                
                20.41f, 52.2f, 14.35f, 39.6f
            };

            Tensor a = CreateTensor(shapeA, dataA);
            Tensor b = CreateTensor(shapeB, dataB);
            Tensor c = CreateTensor();

            a.Dot2D(b, a.Batch, a.Width, b.Height, b.Width, shapeC, c);
            
            Assert.Equal(3, c.Batch);
            Assert.Equal(1, c.Channels);
            Assert.Equal(1, c.Height);
            Assert.Equal(4, c.Width);
            Assert.True(FloatComparison.AreEqual(res, c.Storage.Data));
        }

        [Fact]
        public void Transpose2D()
        {
            Shape shape = new Shape(1, 1, 2, 3);
            float[] data =
            {
                0.2f, 1, 2.8f,
                3, 4.1f, 5
            };
            
            float[] res =
            {
                0.2f, 3,
                1, 4.1f,
                2.8f, 5
            };
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            
            t.Transpose2D(tRes);
            
            Assert.Equal(1, tRes.Batch);
            Assert.Equal(1, tRes.Channels);
            Assert.Equal(3, tRes.Height);
            Assert.Equal(2, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Data));
        }

        [Fact]
        public void Rotate180()
        {
            Shape shape = new Shape(2, 2, 2, 2);
            float[] data =
            {
                0f, 1,
                2, 3,
                
                4, 5,
                6, 7,
                
                8, 9,
                10, 11,
                
                12, 13,
                14, 15
            };
            
            float[] res =
            {
                3f, 2,
                1, 0,
                
                7, 6,
                5, 4,
                
                11, 10,
                9, 8,
                
                15, 14,
                13, 12
            };
            
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            
            t.Rotate180(tRes);
            
            Assert.Equal(2, tRes.Batch);
            Assert.Equal(2, tRes.Channels);
            Assert.Equal(2, tRes.Height);
            Assert.Equal(2, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Data));
        }

        [Fact]
        public void Max()
        {
            Shape shape = new Shape(2, 1, 1, 3);
            float[] data =
            {
                0.2f, 1, 2.8f,
                
                3, 4.1f, 5
            };
            Tensor t = CreateTensor(shape, data);
            Tensor max = CreateTensor();
            
            t.Max(max);
            
            Assert.Equal(2, max.Batch);
            Assert.Equal(1, max.Channels);
            Assert.Equal(1, max.Height);
            Assert.Equal(2, max.Width);
            Assert.Equal(2.8f, max[0]);
            Assert.Equal(2, max[1]);
            Assert.Equal(5f, max[2]);
            Assert.Equal(2, max[3]);
        }

        [Fact]
        public void Sum()
        {
            Shape shapeA = new Shape(1, 2, 2, 2);
            Shape shapeB = new Shape(1, 2, 2, 2);
            
            float[] dataA =
            {
                0.2f, 1,
                2.8f, 3,
                
                4.1f, 5,
                1, 2.2f
            };
            
            float[] dataB =
            {
                0.1f, 2,
                3.5f, 6,
                
                4, 8.8f,
                1.3f, 4.8f
            };
            
            float[] res =
            {
                0.3f, 3,
                6.3f, 9,
                
                8.1f, 13.8f,
                2.3f, 7
            };
            Tensor a = CreateTensor(shapeA, dataA);
            Tensor b = CreateTensor(shapeB, dataB);

            a.Sum(b);
            
            Assert.True(FloatComparison.AreEqual(res, a.Storage.Data));
        }
        
        [Fact]
        public void Fill()
        {
            Shape shape = new Shape(1, 2, 2, 2);
            float[] data =
            {
                0, 0,
                0, 0,
                
                0, 0,
                0, 0
            };
            
            float[] res =
            {
                3, 3,
                3, 3,
                
                3, 3,
                3, 3
            };
            
            Tensor t = CreateTensor(shape, data);
            
            t.Fill(3);
            
            Assert.True(FloatComparison.AreEqual(res, t.Storage.Data));
        }
        
        [Fact]
        public void Im2Col()    
        {
            Shape shape = new Shape(2, 2, 3, 3);
            float[] data =
            {
                //1st batch
                0.2f, 1, 2.8f,
                3, 4.1f, 5,
                1, 2.2f, 4,
                
                1.8f, 5, 8,
                2, 2.1f, 0,
                0.4f, 0.9f, 0,
                
                //2nd batch
                1, 0, 2,
                2.1f, 4.5f, 0,
                3, 8, 1,
                
                2, 3, 4,
                7, 7.2f, 0.1f,
                0, 0, 2
            };
            
            float[] res = 
            { 
                0.2f, 1, 3, 4.1f, 1, 0, 2.1f, 4.5f,
                1, 2.8f, 4.1f, 5, 0, 2, 4.5f, 0,
                3, 4.1f, 1, 2.2f, 2.1f, 4.5f, 3, 8,
                4.1f, 5, 2.2f, 4, 4.5f, 0, 8, 1, 
                1.8f, 5, 2, 2.1f, 2, 3, 7, 7.2f,
                5, 8, 2.1f, 0, 3, 4, 7.2f, 0.1f,
                2, 2.1f, 0.4f, 0.9f, 7, 7.2f, 0, 0,
                2.1f, 0, 0.9f, 0, 7.2f, 0.1f, 0, 2 
            };
            
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            
            t.Im2Col(2, 2, 1, tRes);
            
            Assert.Equal(1, tRes.Batch);
            Assert.Equal(1, tRes.Channels);
            Assert.Equal(8, tRes.Height);
            Assert.Equal(8, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Data));
        }

        [Fact]
        public void Col2Im()
        {
            var shape = new Shape(1, 1, 2, 8);
            var outShape = new Shape(2, 2, 2, 2);

            float[] data =
            {
                4.1f, 5, 1.1f, 2, 5, 8, 2.1f, 2.1f,
                0.1f, 1, 3.5f, 0.05f, 5, 0.8f, 3, 0
            };
            
            float[] res =
            {
                4.1f, 5,
                1.1f, 2,
                
                0.1f, 1,
                3.5f, 0.05f,
                
                5, 8,
                2.1f, 2.1f,
                
                5, 0.8f,
                3, 0 
            };
            
            var t = CreateTensor(shape, data);
            var tRes = CreateTensor();
            
            t.Col2Im(outShape, tRes);
            
            Assert.Equal(2, tRes.Batch);
            Assert.Equal(2, tRes.Channels);
            Assert.Equal(2, tRes.Height);
            Assert.Equal(2, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Data));
        }
        
        [Fact]
        public void Pad()
        {
            Shape shape = new Shape(2, 2, 1, 1);
            float[] data =
            {
                0.2f,
                
                1,
                
                2.8f,
                
                3
            };
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
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Data));
        }
        
        [Fact]
        public void FullyConnectedDx()
        {
            Shape xShape = new Shape(2, 1, 1, 4);
            Shape dyShape = new Shape(2, 1, 1, 2);
            Shape wShape = new Shape(1, 1, 4, 2);
            
            float[] wData = 
            {
                0, 1.5f,
                2, 0.2f,
                5, 2.5f,
                0.1f, 4
            };
            
            float[] xData =
            {
                2, 1.5f, 0.1f, 0.25f,
                
                2, 1.5f, 0.1f, 0.25f
            };
            
            float[] dyData =
            {
                1, 2.5f,
                
                0, 1.2f
            };
            
            float[] dxExp =
            {
                3.75f, 2.5f, 11.25f,
                
                10.1f, 1.8f, 0.24f, 3, 4.8f
            };
            
            Tensor x = CreateTensor(xShape, xData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dx = CreateTensor();
            
            x.FullyConnectedDx(w, dy, CreateTensor(), dx);
            
            Assert.Equal(2, dx.Batch);
            Assert.Equal(1, dx.Channels);
            Assert.Equal(1, dx.Height);
            Assert.Equal(4, dx.Width);
            Assert.True(FloatComparison.AreEqual(dxExp, dx.Storage.Data));
        }        
        
        [Fact]
        public void FullyConnectedDw()
        {
            Shape xShape = new Shape(2, 1, 1, 4);
            Shape dyShape = new Shape(2, 1, 1, 2);
            
            float[] xData = 
            { 
                2, 1.5f, 0.1f, 0.25f,
                
                0, 3.5f, 7.1f, 3.29f
            };
            
            float[] dyData = 
            {                 
                1, 2.5f,
                
                0, 1.2f 
            };
            
            float[] dwExp =
            {
                2, 5,
                1.5f, 7.95f,
                0.1f, 8.77f,
                0.25f, 4.573f
            };
            Tensor x = CreateTensor(xShape, xData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dw = CreateTensor();
            
            x.FullyConnectedDw(dy, CreateTensor(), dw);
            
            Assert.Equal(1, dw.Batch);
            Assert.Equal(1, dw.Channels);
            Assert.Equal(4, dw.Height);
            Assert.Equal(2, dw.Width);
            Assert.True(FloatComparison.AreEqual(dwExp, dw.Storage.Data));
        }
        
        [Fact]
        public void Convolution()
        {
            Shape xShape = new Shape(2, 2, 3, 3);
            Shape wShape = new Shape(2, 2, 2, 2);
            
            float[] xData =
            {
                //1st batch
                0.2f, 1, 2.8f,
                3, 4.1f, 5,
                1, 2.2f, 4,

                1.8f, 5, 8,
                2, 2.1f, 0,
                0.4f, 0.9f, 0,

                //2nd batch
                1, 0, 2,
                2.1f, 4.5f, 0,
                3, 8, 1,

                2, 3, 4,
                7, 7.2f, 0.1f,
                0, 0, 2
            };
            
            float[] wData =
            {
                0.1f, 2, 
                3.5f, 6,
                
                4, 8.8f,
                1.3f, 4.8f,
                
                1, 0,
                5, 0,
                
                0, 3.5f,
                4, 0.5f
            };
            
            float[] res =
            {
                101, 143.18f,
                56.52f, 51.68f,
                
                41.75f, 57.9f,
                17.4f, 18.7f,
                
                112.51f, 76.79f,
                159.07f, 73.73f,
                
                53.6f, 65.35f,
                42.3f, 45.85f
            };
            Tensor x = CreateTensor(xShape, xData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor y = CreateTensor();
            
            Tensor im2ColBuffer = CreateTensor();
            Tensor dotBuffer = CreateTensor();
            
            x.Convolution(w, 1, im2ColBuffer, dotBuffer, y);
            
            Assert.Equal(2, y.Batch);
            Assert.Equal(2, y.Channels);
            Assert.Equal(2, y.Height);
            Assert.Equal(2, y.Width);
            
            Assert.True(FloatComparison.AreEqual(res, y.Storage.Data));
        }
        
        [Fact]
        public void ConvolutionDx()
        {
            Shape xShape = new Shape(2, 2, 3, 3);
            Shape wShape = new Shape(2, 2, 2, 2);
            Shape dyShape = new Shape(2, 2, 2, 2);
            
            float[] xData =
            {
                //1st batch
                0.2f, 1, 2.8f,
                3, 4.1f, 5,
                1, 2.2f, 4,

                1.8f, 5, 8,
                2, 2.1f, 0,
                0.4f, 0.9f, 0,

                //2nd batch
                1, 0, 2,
                2.1f, 4.5f, 0,
                3, 8, 1,

                2, 3, 4,
                7, 7.2f, 0.1f,
                0, 0, 2
            };
            
            float[] wData =
            {
                0.1f, 2,
                3.5f, 6,
                
                4, 8.8f,
                1.3f, 4.8f,
                
                1, 0,
                5, 0,
                
                0, 3.5f,
                4, 0.5f
            };
            
            float[] dyData =
            {
                10, 1.1f,
                6.5f, 1.6f,
                
                4.7f, 7.9f,
                7.4f, 8.7f,
                
                0, 1.9f,
                0.5f, 1.8f,
                
                4.7f, 1,
                7.7f, 0.7f
            };
            
            float[] dxExp =
            {
                5.7f, 28.01f, 2.2f,
                66.55f, 125.21f, 9.8f,
                59.75f, 88.1f, 9.6f,
                
                40f, 108.85f, 37.33f,
                57.8f, 172.88f, 53.76f,
                38.05f, 71.78f, 12.03f,
                
                4.7f, 1.19f, 3.8f,
                31.25f, 13.53f, 15f,
                40.25f, 12.8f, 10.8f,
                
                0f, 24.05f, 20.22f,
                20.8f, 47.37f, 27.91f,
                31.45f, 11.39f, 8.99f
            };

            var w = CreateTensor(wShape, wData);
            var x = CreateTensor(xShape, xData);
            var dy = CreateTensor(dyShape, dyData);
            var dx = CreateTensor();
            
            x.ConvolutionDx(w, dy,  CreateTensor(), CreateTensor(), CreateTensor(), CreateTensor(), CreateTensor(), dx);
            
            Assert.Equal(2, dx.Batch);
            Assert.Equal(2, dx.Channels);
            Assert.Equal(3, dx.Height);
            Assert.Equal(3, dx.Width);
            Assert.True(FloatComparison.AreEqual(dxExp, dx.Storage.Data));
        }
        
        [Fact]
        public void ConvolutionDw()
        {
            Shape xShape = new Shape(2, 2, 3, 3);
            Shape im2ColXShape = new Shape(1, 1, 8, 8);
            Shape wShape = new Shape(2, 2, 2, 2);
            Shape dyShape = new Shape(2, 2, 2, 2);
            
            float[] xData =
            {
                //1st batch
                0.2f, 1, 2.8f,
                3, 4.1f, 5,
                1, 2.2f, 4,

                1.8f, 5, 8,
                2, 2.1f, 0,
                0.4f, 0.9f, 0,

                //2nd batch
                1, 0, 2,
                2.1f, 4.5f, 0,
                3, 8, 1,

                2, 3, 4,
                7, 7.2f, 0.1f,
                0, 0, 2
            };
            
            float[] im2ColXData =
            {
                0.2f, 1, 3, 4.1f, 1, 0, 2.1f, 4.5f,
                1, 2.8f, 4.1f, 5, 0, 2, 4.5f, 0,
                3, 4.1f, 1, 2.2f, 2.1f, 4.5f, 3, 8,
                4.1f, 5, 2.2f, 4, 4.5f, 0, 8, 1, 
                1.8f, 5, 2, 2.1f, 2, 3, 7, 7.2f,
                5, 8, 2.1f, 0, 3, 4, 7.2f, 0.1f,
                2, 2.1f, 0.4f, 0.9f, 7, 7.2f, 0, 0,
                2.1f, 0, 0.9f, 0, 7.2f, 0.1f, 0, 2 
            };
            
            float[] wData =
            {
                0.1f, 2,
                3.5f, 6,
                
                4, 8.8f,
                1.3f, 4.8f,
                
                1, 0,
                5, 0,
                
                0, 3.5f,
                4, 0.5f
            };
            
            float[] dyData =
            {
                10, 1.1f,
                6.5f, 1.6f,
                
                4.7f, 7.9f,
                7.4f, 8.7f,
                
                0, 1.9f,
                0.5f, 1.8f,
                
                4.7f, 1,
                7.7f, 0.7f
            };
            
            float[] dwExp =
            {
                38.31f, 53.78f,
                68.98f, 73f,  
                
                62.02f, 83.83f,
                40.03f, 30.64f,
                
                90.73f, 137.31f, 
                116.1f, 193.3f,
                
                152.37f, 175.85f,
                76.88f, 51.87f
            };
            
            Tensor x = CreateTensor(xShape, xData);
            Tensor im2ColX = CreateTensor(im2ColXShape, im2ColXData);
            Tensor w = CreateTensor(wShape, wData);
            Tensor dy = CreateTensor(dyShape, dyData);
            Tensor dw = CreateTensor();
            
            Tensor dy2DBuffer = CreateTensor();
            Tensor dot2DBuffer = CreateTensor();
            
            x.ConvolutionDw(w, dy, dy2DBuffer, dot2DBuffer, im2ColX, dw);
            
            Assert.Equal(2, dw.Batch);
            Assert.Equal(2, dw.Channels);
            Assert.Equal(2, dw.Height);
            Assert.Equal(2, dw.Width);
            Assert.True(FloatComparison.AreEqual(dwExp, dw.Storage.Data));
        }
        
        [Fact]
        public void MaxPool()
        {
            Shape shape = new Shape(2, 2, 3, 3);
            float[] data =
            {
                0.2f, 1, 2.8f,
                3, 4.1f, 5,
                1, 2.2f, 4,

                1.8f, 5, 8,
                2, 2.1f, 0,
                0.4f, 0.9f, 0,

                1, 0, 2,
                2.1f, 4.5f, 0,
                3, 8, 1,

                2, 3, 4,
                7, 7.2f, 0.1f,
                0, 0, 2
            };
            
            float[] res =
            {
                4.1f, 5,
                4.1f, 5,
                
                5, 8,
                2.1f, 2.1f,
                
                4.5f, 4.5f,
                8, 8,
                
                7.2f, 7.2f,
                7.2f, 7.2f
            };
            
            float[] maxInd =
            {
                4, 5, 4, 5, 10, 11, 13, 13, 22, 22, 25, 25, 31, 31, 31, 31
            };
            
            Tensor t = CreateTensor(shape, data);
            Tensor tRes = CreateTensor();
            Tensor tMaxInd = CreateTensor();
            
            t.MaxPool(2, 1, tRes, tMaxInd);
            
            Assert.Equal(2, tRes.Batch);
            Assert.Equal(2, tRes.Channels);
            Assert.Equal(2, tRes.Height);
            Assert.Equal(2, tRes.Width);
            Assert.True(FloatComparison.AreEqual(res, tRes.Storage.Data));
            Assert.True(FloatComparison.AreEqual(maxInd, tMaxInd.Storage.Data));
        }
        
        [Fact]
        public void MaxPoolDx()    
        {
            Shape xShape = new Shape(1, 2, 3, 3);
            Shape dyShape = new Shape(1, 2, 2, 2);
            Shape maxIndShape = new Shape(1, 1, 1, 8);
            float[] x = new float[18];
            float[] dy =
            {
                4.1f, 5,
                4.1f, 5,
                
                5, 8,
                2.1f, 2.1f
            };
            
            float[] maxInd = { 4, 5, 4, 5, 10, 11, 13, 13 };
            float[] dx = new float[18];
            dx[4] = 4.1f;
            dx[5] = 5;
            dx[10] = 5;
            dx[11] = 8;
            dx[13] = 2.1f;
            
            Tensor xTensor = CreateTensor(xShape, x);
            Tensor dyTensor = CreateTensor(dyShape, dy);
            Tensor maxTensor = CreateTensor(maxIndShape, maxInd);
            Tensor dxTensor = CreateTensor();
            
            xTensor.MaxPoolDx(dyTensor, maxTensor, dxTensor);
            
            Assert.Equal(1, dxTensor.Batch);
            Assert.Equal(2, dxTensor.Channels);
            Assert.Equal(3, dxTensor.Height);
            Assert.Equal(3, dxTensor.Width);
            Assert.True(FloatComparison.AreEqual(dx, dxTensor.Storage.Data));
        }

        [Fact]
        public void Softmax()
        {
            Shape shape = new Shape(2, 1, 1, 5);
            float[] xData =
            {
                1, 4.5f, 0, 2, 3.1f,
                0, 3.5f, 0.8f, 0.25f, 2.5f
            };
            
            float[] yExp =
            {
                0.022f, 0.7299f, 0.0081f, 0.0599f, 0.1799f,
                0.02f, 0.66486f, 0.044682f, 0.02577f, 0.24459f
            };
            
            Tensor x = CreateTensor(shape, xData);
            Tensor y = CreateTensor();
            Tensor max = CreateTensor();
            x.Softmax(y, max);

            Assert.True(FloatComparison.AreEqual(yExp, y.Storage.Data));
        }
        
        [Fact]
        public void SoftmaxDx()
        {
            Shape yShape = new Shape(2, 1, 1, 5);
            Shape dyShape = new Shape(2, 1, 1, 5);
            float[] yData =
            {
                1, 4.5f, 0, 2, 3.1f,
                0, 1.5f, 2.9f, 1, 0.2f
            };
            
            float[] dyData =
            {
                0.5f, 3, 0.4f, 1, 2.5f, 
                0.01f, 1.5f, 0, 3.1f, 0.28f
            };
            
            float[] dxExp =
            {
                -23.25f, -93.375f, 0, -45.5f, -65.875f,
                0, -5.859f, -15.6774f, -2.306f, -1.0252f
            };
            
            Tensor dx = CreateTensor();
            Tensor y = CreateTensor(yShape, yData);
            Tensor dy = CreateTensor(dyShape, dyData);
            
            y.SoftmaxDx(dy, dx);
            
            Assert.True(FloatComparison.AreEqual(dxExp, dx.Storage.Data));
        }

        [Fact]
        public void ToFlatten()
        {
            Shape shape = new Shape(2, 2, 2, 2);
            float[] data =
            {
                1, 2,
                3, 4,
                
                5, 6,
                7, 8,
                
                9, 10,
                11, 12,
                
                13, 14,
                15, 16
            };
            
            Tensor x = CreateTensor(shape, data);
            Tensor y = CreateTensor();
            
            x.ToFlatten(y);
            
            Assert.Equal(2, y.Batch);
            Assert.Equal(1, y.Channels);
            Assert.Equal(1, y.Height);
            Assert.Equal(8, y.Width);
            Assert.True(FloatComparison.AreEqual(data, y.Storage.Data));
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
            Assert.True(FloatComparison.AreEqual(dyData, dx.Storage.Data));
        }

        [Fact]
        public void CrossEntropy()
        {
            var oData = new [] { 0.5f, 1, 2, 0.1f, 1.2f, 1.4f, 3, 1, 2.5f, 1.1f };
            var tData = new[] { 1, 0.2f, 2.3f, 0.3f, 2, 2, 0.5f, 1, 2.8f, 0.05f };
            
            var shape = new Shape(2, 1, 1, 5);
            
            var o = CreateTensor(shape, oData);
            var t = CreateTensor(shape, tData);
            var loss = CreateTensor();
            
            o.Loss(t, new CrossEntropy(), loss);
            
            Assert.Equal(2, loss.Size);
            Assert.Equal(2, loss.Batch);
            Assert.True(FloatComparison.AreEqual(-0.575f, loss[0]));
            Assert.True(FloatComparison.AreEqual(-3.792f, loss[1]));
        }

        [Fact]
        public void MeanSquaredError()
        {
            var oData = new [] { 0.5f, 1, 2, 0.1f, 1.2f, 1.4f, 3, 1, 2.5f, 1.1f };
            var tData = new[] { 1, 0.2f, 2.3f, 0.3f, 2, 2, 0.5f, 1, 2.8f, 0.05f };
            
            var shape = new Shape(2, 1, 1, 5);
            
            var o = CreateTensor(shape, oData);
            var t = CreateTensor(shape, tData);
            var loss = CreateTensor();
            
            o.Loss(t, new MeanSquaredError(), loss);
            
            Assert.Equal(2, loss.Size);
            Assert.Equal(2, loss.Batch);
            Assert.True(FloatComparison.AreEqual(0.331f, loss[0]));
            Assert.True(FloatComparison.AreEqual(1.56f, loss[1]));
        }
    }
}
