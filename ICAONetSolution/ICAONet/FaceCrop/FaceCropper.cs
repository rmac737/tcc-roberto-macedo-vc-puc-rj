using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

namespace ICAONet.FaceCrop
{
    public class FaceCropper
    {
        const int OUT_W = 480; // largura final
        const int OUT_H = 640; // altura final

        readonly (float min, float max) FaceWidthFrac = (0.50f, 0.70f);
        readonly (float min, float max) FaceHeightFrac = (0.60f, 0.80f);

        public async Task<Mat> CropFace(Mat bgr, Point2f[] lm)
        {
            return await Task.Run(() =>
            {
                if (bgr.Empty()) throw new ArgumentException("Imagem vazia.");
                if (lm == null || lm.Length < 468) throw new ArgumentException("São necessários 468 pontos.");

                // Bounding box do rosto
                float minX = lm.Min(p => p.X);
                float maxX = lm.Max(p => p.X);
                float minY = lm.Min(p => p.Y);
                float maxY = lm.Max(p => p.Y);

                float faceW = maxX - minX;
                float faceH = maxY - minY;

                float cx = (minX + maxX) / 2f;
                float cy = (minY + maxY) / 2f;

                // Escala mínima/máxima para largura
                float sW_min = FaceWidthFrac.min * OUT_W / faceW;
                float sW_max = FaceWidthFrac.max * OUT_W / faceW;

                // Escala mínima/máxima para altura
                float sH_min = FaceHeightFrac.min * OUT_H / faceH;
                float sH_max = FaceHeightFrac.max * OUT_H / faceH;

                float s_min = Math.Max(sW_min, sH_min);
                float s_max = Math.Min(sW_max, sH_max);

                float s = (s_min <= s_max) ? s_min : Math.Min(s_min, s_max);

                // ROI no espaço da imagem original
                float roiW = OUT_W / s;
                float roiH = OUT_H / s;

                var roi = new Rect2f(cx - roiW / 2f, cy - roiH / 2f, roiW, roiH);

                // Garantir que está dentro (padding se necessário)
                Mat padded;
                Rect roiOnPadded;
                PadIfNeeded(bgr, roi, out padded, out roiOnPadded);

                // Crop + resize final
                var cropped = new Mat(padded, roiOnPadded);
                Mat output = new Mat();
                Cv2.Resize(cropped, output, new Size(OUT_W, OUT_H), 0, 0, InterpolationFlags.Area);

                return output;
            });
        }
        private void PadIfNeeded(Mat src, Rect2f roiF, out Mat padded, out Rect roiIntOnPadded)
        {
            int left = (int)Math.Ceiling(Math.Max(0f, -roiF.X));
            int top = (int)Math.Ceiling(Math.Max(0f, -roiF.Y));
            int right = (int)Math.Ceiling(Math.Max(0f, (roiF.X + roiF.Width) - src.Cols));
            int bottom = (int)Math.Ceiling(Math.Max(0f, (roiF.Y + roiF.Height) - src.Rows));

            if (left > 0 || top > 0 || right > 0 || bottom > 0)
            {
                Mat tmp = new Mat();
                Cv2.CopyMakeBorder(src, tmp, top, bottom, left, right, BorderTypes.Reflect101);
                padded = tmp;

                var x = roiF.X + left;
                var y = roiF.Y + top;
                roiIntOnPadded = new Rect(
                    (int)Math.Floor(x),
                    (int)Math.Floor(y),
                    (int)Math.Round(roiF.Width),
                    (int)Math.Round(roiF.Height)
                );
            }
            else
            {
                padded = src;
                roiIntOnPadded = new Rect(
                    (int)Math.Floor(roiF.X),
                    (int)Math.Floor(roiF.Y),
                    (int)Math.Round(roiF.Width),
                    (int)Math.Round(roiF.Height)
                );
            }
        }
    }
}
