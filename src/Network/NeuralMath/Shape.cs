﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace Network.NeuralMath
{
    public class Shape
    {
        /* NCHW Format
           [0] - Number of images
           [1] - Number of channels
           [2] - Height of image
           [3] - Width of image
        */
        
        public int[] Dimensions { get; }
        public int Size { get; }

        public Shape(int b, int c, int h, int w)
        {
            Dimensions = new[] { b, c, h, w };
            Size = b * c * w * h;
        }

        public static bool operator ==(Shape a, Shape b)
        {
            if (a is null || b is null)
                return false;
            
            return a.Dimensions.SequenceEqual(b.Dimensions);
        }

        public static bool operator !=(Shape a, Shape b)
        {
            if (a is null || b is null)
                return true;
            
            return !a.Dimensions.SequenceEqual(b.Dimensions);
        }

        public override string ToString()
        {
            string str = Dimensions[0].ToString();
            for (int i = 1; i < Dimensions.Length; i++)
            {
                str += " x " + Dimensions[i];
            }

            return str; 
        }

        public Shape GetCopy()
        {
            return new Shape(Dimensions[0], Dimensions[1], Dimensions[2], Dimensions[3]);
        }
    }
}