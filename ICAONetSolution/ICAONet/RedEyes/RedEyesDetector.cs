using Compunet.YoloSharp;
using ICAONet.Enums;
using OpenCvSharp;
using System;
using System.Linq;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace ICAONet.RedEyes
{
    public class RedEyesDetector
    {
        YoloPredictor? yoloPredictor;
        private Utils _utils;

        // Contornos padrão do FaceMesh (olho completo – superior + inferior)
        // Referência comum (MediaPipe): 
        // Left: 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246
        // Right: 263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466
        readonly int[] LeftEyeIdx = { 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 };
        readonly int[] RightEyeIdx = { 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466 };

        public RedEyesDetector()
        {
            _utils = new Utils();
            yoloPredictor = new YoloPredictor(_utils.LoadResource("ICAONet.RedEyes.red_no_red.onnx"));
        }

        public async Task<(bool, string)> Predict(Mat imagem, Point2f[] faceLocation)
        {

            //Crop Both Eyes
            var (matRoi, rect) =  CropBothEyes(imagem, faceLocation, 2.5);

            Mat resized = new Mat();
            Cv2.Resize(matRoi, resized, new Size(640, 640), 0, 0, InterpolationFlags.Linear);

            // Run model
            var result = await yoloPredictor!.ClassifyAsync(resized.ToBytes());

            foreach (var pred in result.ToList())
            {
                if (pred.Confidence >= 0.50)
                {
                    if (pred.Name.Name.Contains("nao"))
                    {
                        return (true, $"{EnumICAOResult.PASSED}. No Red Eyes detected.");
                    }
                    else
                    {
                        return (false, $"{EnumICAOResult.NOT_PASSED}. Red Eyes detected.");
                    }
                }
            }

            // or
            return (false, "");
        }

        #region "Private Methods"
        public (Mat roi, Rect rect) CropBothEyes(Mat bgr, Point2f[] lm468, double marginScale = 1.35)
        {
            var rect = RectFromPoints(bgr.Size(), lm468, LeftEyeIdx.Concat(RightEyeIdx).ToArray(), marginScale);
            if (rect.Width <= 0 || rect.Height <= 0) return (new Mat(), new Rect());
            return (new Mat(bgr, rect).Clone(), rect);
        }

        // Recorta CADA olho (esquerdo e direito) separadamente.
        public (Mat left, Rect leftRect, Mat right, Rect rightRect)
        CropEachEye(Mat bgr, Point2f[] lm468, double marginScale = 1.25)
        {
            var lrect = RectFromPoints(bgr.Size(), lm468, LeftEyeIdx, marginScale);
            var rrect = RectFromPoints(bgr.Size(), lm468, RightEyeIdx, marginScale);

            var left = (lrect.Width > 0 && lrect.Height > 0) ? new Mat(bgr, lrect).Clone() : new Mat();
            var right = (rrect.Width > 0 && rrect.Height > 0) ? new Mat(bgr, rrect).Clone() : new Mat();

            return (left, lrect, right, rrect);
        }

        public (Mat roi, Rect rect) CropBothEyesRectified(Mat bgr, Point2f[] lm468, double marginScale = 1.35)
        {
            // ângulo médio entre cantos dos olhos
            double aLeft = AngleDeg(lm468[33], lm468[133]);
            double aRight = AngleDeg(lm468[362], lm468[263]);
            double angle = (aLeft + aRight) * 0.5;

            var center = new Point2f(bgr.Cols / 2f, bgr.Rows / 2f);
            var M = Cv2.GetRotationMatrix2D(center, angle, 1.0);

            // rotaciona imagem
            using var rot = new Mat();
            Cv2.WarpAffine(bgr, rot, M, bgr.Size(), InterpolationFlags.Linear, BorderTypes.Replicate);

            // rotaciona também os pontos necessários e calcula o retângulo sobre eles
            var allIdx = LeftEyeIdx.Concat(RightEyeIdx).ToArray();
            var rotatedPts = allIdx.Select(i => TransformPt(lm468[i], M)).ToArray();

            var rect = RectFromRawPoints(rot.Size(), rotatedPts, marginScale);
            if (rect.Width <= 0 || rect.Height <= 0) return (new Mat(), new Rect());
            return (new Mat(rot, rect).Clone(), rect);
        }

        // --- Privado ---
        Rect RectFromPoints(Size imgSize, Point2f[] lm, int[] idxs, double scale)
        {
            var pts = idxs.Select(i => lm[i]).ToArray();
            return RectFromRawPoints(imgSize, pts, scale);
        }

        Rect RectFromRawPoints(Size imgSize, Point2f[] pts, double scale)
        {
            if (pts.Length == 0) return new Rect(0, 0, 0, 0);

            float minX = pts.Min(p => p.X), maxX = pts.Max(p => p.X);
            float minY = pts.Min(p => p.Y), maxY = pts.Max(p => p.Y);

            // retângulo justo
            int x = (int)Math.Floor(minX);
            int y = (int)Math.Floor(minY);
            int w = (int)Math.Ceiling(maxX - minX);
            int h = (int)Math.Ceiling(maxY - minY);
            if (w <= 0) w = 1;
            if (h <= 0) h = 1;

            var r = new Rect(x, y, w, h);
            r = ExpandRect(r, scale, imgSize);
            return r;
        }

        Rect ExpandRect(Rect r, double scale, Size limit)
        {
            double cx = r.X + r.Width / 2.0;
            double cy = r.Y + r.Height / 2.0;
            int w = (int)Math.Round(r.Width * scale);
            int h = (int)Math.Round(r.Height * scale);
            int x = (int)Math.Round(cx - w / 2.0);
            int y = (int)Math.Round(cy - h / 2.0);

            // clip nos limites da imagem
            x = Math.Max(0, x);
            y = Math.Max(0, y);
            if (x + w > limit.Width) w = limit.Width - x;
            if (y + h > limit.Height) h = limit.Height - y;
            if (w < 0) w = 0; if (h < 0) h = 0;

            return new Rect(x, y, w, h);
        }

        double AngleDeg(Point2f a, Point2f b) =>
            Math.Atan2(b.Y - a.Y, b.X - a.X) * 180.0 / Math.PI;

        Point2f TransformPt(Point2f p, Mat M)
        {
            // M é 2x3: [m00 m01 m02; m10 m11 m12]
            double m00 = M.At<double>(0, 0), m01 = M.At<double>(0, 1), m02 = M.At<double>(0, 2);
            double m10 = M.At<double>(1, 0), m11 = M.At<double>(1, 1), m12 = M.At<double>(1, 2);
            float x = (float)(m00 * p.X + m01 * p.Y + m02);
            float y = (float)(m10 * p.X + m11 * p.Y + m12);
            return new Point2f(x, y);
        }
        #endregion
    }
}
