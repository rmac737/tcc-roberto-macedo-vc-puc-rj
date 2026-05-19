using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.Helpers
{
    public static class PixelationHelper
    {
        public static (double avg, double std) DiffStats(Mat gray, bool horizontal)
        {
            Mat a, b;
            if (horizontal)
            {
                a = new Mat(gray, new Rect(1, 0, gray.Cols - 1, gray.Rows));
                b = new Mat(gray, new Rect(0, 0, gray.Cols - 1, gray.Rows));
            }
            else
            {
                a = new Mat(gray, new Rect(0, 1, gray.Cols, gray.Rows - 1));
                b = new Mat(gray, new Rect(0, 0, gray.Cols, gray.Rows - 1));
            }

            using var diff = new Mat();
            Cv2.Absdiff(a, b, diff);

            Cv2.MeanStdDev(diff, out Scalar mean, out Scalar stddev);
            return (mean.Val0, stddev.Val0);
        }
    }
}
