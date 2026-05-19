using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

namespace ICAONet.FaceDetection
{
    public class HaarCascadeDetection
    {
        string haarPath = @"C:\github\iso-iec-19794-5\ICAONet.AppTest\bin\Debug\net8.0\haarcascade_frontalface_default.xml";
        public Rect DetectFirstFaceWithHaar(Mat bgr)
        {
            using var gray = new Mat();
            Cv2.CvtColor(bgr, gray, ColorConversionCodes.BGR2GRAY);
            using var cascade = new CascadeClassifier(haarPath);
            var faces = cascade.DetectMultiScale(gray, 1.1, 3, HaarDetectionTypes.ScaleImage, new Size(80, 80));
            return faces.Length > 0 ? faces[0] : new Rect();
        }
    }
}
