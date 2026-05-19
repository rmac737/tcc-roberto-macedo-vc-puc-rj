using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace ICAONet.Helpers
{
    public class PosterizationHelper
    {
        public static (bool Posterized, string Reason) Check(
            byte[] imageBytes,
            int resizeLongEdge = 640,
            int quantStep = 16,              // ex.: 12..20
            double occupancyMax = 0.35,      // fração de bins ocupados
            double entropyMaxBits = 4.8,     // H em bits
            double avgRunLenMin = 6.0,       // runs longos após quantização
            double reQuantMaeMax = 5.0,      // MAE entre original e reconst.
            double zeroGradMinRatio = 0.25,  // pixels com |∇| ~ 0 (3x3)
            bool keepAspect = true
        )
        {
            using var img0 = Cv2.ImDecode(imageBytes, ImreadModes.AnyColor);
            if (img0.Empty()) throw new ArgumentException("Imagem inválida.");

            using var gray = ToGrayU8(img0);
            using var grayRes = ResizeIfNeeded(gray, resizeLongEdge, keepAspect);

            using var q = QuantizeLUT(grayRes, quantStep); // bins 0..(255/step)

            var (occupancy, entropyBits, levels, spacingMedian) = PaletteStats(q, quantStep);
            var avgRunLen = AvgRunLength(q);
            var reqMae = ReQuantMae(grayRes, q, quantStep);
            var zeroGradRatio = ZeroGradRatio(grayRes, thr: 1.5); // 1.0~2.0 funciona bem

            int votes = 0;
            var reasons = new List<string>(6);

            if (entropyBits <= entropyMaxBits) { votes++; reasons.Add($"lowEntropy H={entropyBits:0.00}≤{entropyMaxBits}"); }
            if (avgRunLen >= avgRunLenMin) { votes++; reasons.Add($"longRuns {avgRunLen:0.00}≥{avgRunLenMin}"); }
            if (reqMae <= reQuantMaeMax) { votes++; reasons.Add($"reQuantMAE {reqMae:0.00}≤{reQuantMaeMax}"); }
            if (zeroGradRatio >= zeroGradMinRatio) { votes++; reasons.Add($"zeroGrad {zeroGradRatio:0.00}≥{zeroGradMinRatio}"); }
            if (occupancy <= occupancyMax) { votes++; reasons.Add($"sparsePal {occupancy:0.00}≤{occupancyMax}"); }

            bool posterized = votes >= 3;

            string reason =
                $"posterized={posterized} votes={votes} | " +
                $"H={entropyBits:0.00}, runs={avgRunLen:0.00}, MAE={reqMae:0.00}, zero∇={zeroGradRatio:0.00}, occ={occupancy:0.00}, dMed={spacingMedian:0.00} (step={quantStep}) | " +
                string.Join(" ; ", reasons);

            return (posterized, reason);
        }

        // ---------- Helpers ----------
        private static Mat ToGrayU8(Mat src)
        {
            if (src.Channels() == 1 && src.Type() == MatType.CV_8U) return src.Clone();
            var g = new Mat();
            if (src.Channels() > 1) Cv2.CvtColor(src, g, ColorConversionCodes.BGR2GRAY);
            else g = src.Clone();
            if (g.Type() != MatType.CV_8U) { var t = new Mat(); g.ConvertTo(t, MatType.CV_8U); g.Dispose(); return t; }
            return g;
        }

        private static Mat ResizeIfNeeded(Mat src, int longEdge, bool keepAspect)
        {
            if (longEdge <= 0) return src.Clone();
            int w = src.Cols, h = src.Rows, me = Math.Max(w, h);
            if (me <= longEdge) return src.Clone();
            OpenCvSharp.Size d;
            if (keepAspect)
            {
                double r = longEdge / (double)me;
                d = new OpenCvSharp.Size(Math.Max(1, (int)Math.Round(w * r)), Math.Max(1, (int)Math.Round(h * r)));
            }
            else d = new OpenCvSharp.Size(longEdge, longEdge);
            var dst = new Mat();
            Cv2.Resize(src, dst, d, 0, 0, InterpolationFlags.Area);
            return dst;
        }

        // Quantização exata por LUT (equivalente a floor(gray/step))
        private static Mat QuantizeLUT(Mat grayU8, int step)
        {
            step = Math.Clamp(step, 1, 255);
            var lut = new Mat(1, 256, MatType.CV_8U);
            unsafe
            {
                byte* p = (byte*)lut.DataPointer;
                for (int v = 0; v < 256; v++) p[v] = (byte)(v / step);
            }
            var dst = new Mat();
            Cv2.LUT(grayU8, lut, dst);
            lut.Dispose();
            return dst;
        }

        private static (double occupancy, double entropyBits, List<int> levels, double spacingMedian)
            PaletteStats(Mat q, int step)
        {
            int bins = 255 / step + 1;
            var counts = new long[bins];
            var idx = q.GetGenericIndexer<byte>();
            for (int y = 0; y < q.Rows; y++)
                for (int x = 0; x < q.Cols; x++)
                    counts[idx[y, x]]++;

            long total = counts.Sum();
            var levels = new List<int>();
            double H = 0.0;
            for (int i = 0; i < bins; i++)
            {
                if (counts[i] == 0) continue;
                levels.Add(i);
                double p = (double)counts[i] / total;
                H += -p * (Math.Log(p) / Math.Log(2.0));
            }

            double spacingMedian = 0.0;
            if (levels.Count >= 2)
            {
                levels.Sort();
                var diffs = new List<int>(levels.Count - 1);
                for (int i = 1; i < levels.Count; i++) diffs.Add(levels[i] - levels[i - 1]);
                diffs.Sort();
                spacingMedian = diffs[diffs.Count / 2];
            }

            return (levels.Count / (double)bins, H, levels, spacingMedian);
        }

        private static double AvgRunLength(Mat q)
        {
            double h = AvgRunLengthDir(q, true);
            double v = AvgRunLengthDir(q, false);
            return 0.5 * (h + v);
        }
        private static double AvgRunLengthDir(Mat q, bool horizontal)
        {
            long runSum = 0, runCount = 0;
            var idx = q.GetGenericIndexer<byte>();

            if (horizontal)
            {
                for (int y = 0; y < q.Rows; y++)
                {
                    int run = 1;
                    for (int x = 1; x < q.Cols; x++)
                    {
                        if (idx[y, x] == idx[y, x - 1]) run++;
                        else { runSum += run; runCount++; run = 1; }
                    }
                    runSum += run; runCount++;
                }
            }
            else
            {
                for (int x = 0; x < q.Cols; x++)
                {
                    int run = 1; byte prev = idx[0, x];
                    for (int y = 1; y < q.Rows; y++)
                    {
                        byte cur = idx[y, x];
                        if (cur == prev) run++;
                        else { runSum += run; runCount++; run = 1; prev = cur; }
                    }
                    runSum += run; runCount++;
                }
            }
            return runCount > 0 ? (double)runSum / runCount : 0.0;
        }

        // |∇|≈0: regiões chapadas (posterização aumenta essa fração)
        private static double ZeroGradRatio(Mat grayU8, double thr = 1.5)
        {
            using var gx16 = new Mat(); using var gy16 = new Mat();
            Cv2.Sobel(grayU8, gx16, MatType.CV_16S, 1, 0, 3);
            Cv2.Sobel(grayU8, gy16, MatType.CV_16S, 0, 1, 3);

            using var gx = new Mat(); using var gy = new Mat();
            gx16.ConvertTo(gx, MatType.CV_32F);
            gy16.ConvertTo(gy, MatType.CV_32F);

            using var mag = new Mat();
            Cv2.Magnitude(gx, gy, mag);

            var idx = mag.GetGenericIndexer<float>();
            int flat = 0, total = mag.Rows * mag.Cols;
            for (int y = 0; y < mag.Rows; y++)
                for (int x = 0; x < mag.Cols; x++)
                    if (idx[y, x] <= thr) flat++;

            return total > 0 ? (double)flat / total : 0.0;
        }

        // MAE entre original e reconstrução (q*step + step/2)
        private static double ReQuantMae(Mat grayU8, Mat q, int step)
        {
            using var q32 = new Mat(); q.ConvertTo(q32, MatType.CV_32F);
            using var recon = new Mat();
            Cv2.Multiply(q32, step, recon);
            Cv2.Add(recon, step / 2.0, recon);

            using var g32 = new Mat(); grayU8.ConvertTo(g32, MatType.CV_32F);
            using var diff = new Mat();
            Cv2.Absdiff(g32, recon, diff);

            var m = Cv2.Mean(diff);
            return m.Val0; // MAE
        }
    }
}
