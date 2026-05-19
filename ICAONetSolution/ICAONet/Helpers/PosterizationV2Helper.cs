using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.Helpers
{
    public static class PosterizationV2Helper
    {
        public static (bool Posterized, string Reason) Check(
        byte[] imageBytes,
        int resizeLongEdge = 640,
        int[] quantSteps = null,          // multi-escala p/ paleta/platôs
        // thresholds (ajuste fino abaixo)
        double entropyMaxBits = 4.2,
        double reQuantMaeMax = 3.8,
        double minOccSparse = 0.28,     // ocupação mínima p/ considerar “esparsa” (usamos o MIN entre steps)
        double topBinsMassMin = 0.90,     // massa dos top bins do histograma
        double plateauCoverMin = 0.45,     // fração de área coberta pelos maiores platôs
        double zeroGradMin = 0.15,     // fração de |∇|≈0
        bool keepAspect = true
    )
        {
            quantSteps ??= new[] { 8, 12, 16, 20 };

            using var img0 = Cv2.ImDecode(imageBytes, ImreadModes.AnyColor);
            if (img0.Empty()) throw new ArgumentException("Imagem inválida.");

            using var gray = ToGrayU8(img0);
            using var g = ResizeIfNeeded(gray, resizeLongEdge, keepAspect);

            // 1) Histograma “agrupado” (bins de 4 níveis) → massa dos TOP-M
            double topMass = TopBinsMass(g, group: 4, topM: 8); // 64 bins grupados, pega top-8

            // 2) Zero-gradient ratio (regiões chapadas)
            double zeroGrad = ZeroGradRatio(g, thr: 1.5);

            // 3) Re-quantização (MAE) em um step base (16) — robusto para pôsters
            using var q16 = QuantizeLUT(g, 16);
            double mae16 = ReQuantMae(g, q16, 16);

            // 4) Paleta multi-escala + 5) Platôs grandes (componentes conectados por nível)
            var occList = new List<double>();
            var plateauList = new List<double>();
            var entList = new List<double>();
            foreach (var s in quantSteps)
            {
                using var q = QuantizeLUT(g, s);
                var (occ, Hbits, _, _) = PaletteStats(q, s);
                occList.Add(occ);
                entList.Add(Hbits);
                plateauList.Add(PlateauCoverage(q, takeTopLabels: 6, perLabelTop: 3)); // top-6 níveis, 3 maiores platôs por nível
            }

            double occMin = occList.Min();            // usamos o “melhor caso” p/ detectar esparsidade
            double plateauMax = plateauList.Max();        // e o “pior caso” p/ detectar platôs grandes
            double Hmin = entList.Min();            // menor entropia entre escalas

            // ----- Votação -----
            int votes = 0;
            var reasons = new List<string>(8);

            if (topMass >= topBinsMassMin) { votes++; reasons.Add($"topBinsMass {topMass:0.00}≥{topBinsMassMin}"); }
            if (plateauMax >= plateauCoverMin) { votes++; reasons.Add($"plateauCover {plateauMax:0.00}≥{plateauCoverMin}"); }
            if (occMin <= minOccSparse) { votes++; reasons.Add($"sparsePal (minOcc) {occMin:0.00}≤{minOccSparse}"); }
            if (Hmin <= entropyMaxBits) { votes++; reasons.Add($"lowEntropy (min) {Hmin:0.00}≤{entropyMaxBits}"); }
            if (mae16 <= reQuantMaeMax) { votes++; reasons.Add($"reQuantMAE {mae16:0.00}≤{reQuantMaeMax}"); }
            if (zeroGrad >= zeroGradMin) { votes++; reasons.Add($"zeroGrad {zeroGrad:0.00}≥{zeroGradMin}"); }

            bool posterized = votes >= 3;

            string reason =
                $"posterized={posterized} votes={votes} | " +
                $"topMass={topMass:0.00}, plateauMax={plateauMax:0.00}, occMin={occMin:0.00}, Hmin={Hmin:0.00}, MAE16={mae16:0.00}, zero∇={zeroGrad:0.00} " +
                $"| steps=[{string.Join(",", quantSteps)}] | " + string.Join(" ; ", reasons);

            return (posterized, reason);
        }

        // ----------------- Helpers -----------------
        private static Mat ToGrayU8(Mat src)
        {
            if (src.Channels() == 1 && src.Type() == MatType.CV_8U) return src.Clone();
            var g = new Mat();
            if (src.Channels() > 1) Cv2.CvtColor(src, g, ColorConversionCodes.BGR2GRAY);
            else g = src.Clone();
            if (g.Type() != MatType.CV_8U)
            {
                var t = new Mat();
                g.ConvertTo(t, MatType.CV_8U);
                g.Dispose();
                return t;
            }
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

        // LUT para floor(gray/step)
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

        // histograma 0..255 → agrupa de "group" em "group" (ex.: 4) e soma os top-M
        private static double TopBinsMass(Mat grayU8, int group = 4, int topM = 8)
        {
            int groups = (256 + group - 1) / group;
            var h = new long[groups];
            var idx = grayU8.GetGenericIndexer<byte>();
            for (int y = 0; y < grayU8.Rows; y++)
                for (int x = 0; x < grayU8.Cols; x++)
                    h[idx[y, x] / group]++;

            long total = h.Sum();
            Array.Sort(h); Array.Reverse(h);
            long topSum = h.Take(Math.Min(topM, h.Length)).Sum();
            return total > 0 ? (double)topSum / total : 0.0;
        }

        // fração de pixels com |∇| ~ 0 (Sobel 3x3)
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

        // cobertura por "platôs" (componentes conectados) – soma dos maiores
        private static double PlateauCoverage(Mat q, int takeTopLabels = 6, int perLabelTop = 3)
        {
            // conta labels mais comuns
            int maxLabel = 0;
            var counts = new Dictionary<byte, int>();
            var idx = q.GetGenericIndexer<byte>();
            for (int y = 0; y < q.Rows; y++)
                for (int x = 0; x < q.Cols; x++)
                {
                    byte v = idx[y, x];
                    if (!counts.TryAdd(v, 1)) counts[v]++;
                    if (v > maxLabel) maxLabel = v;
                }

            var topLabels = counts.OrderByDescending(kv => kv.Value)
                                  .Take(takeTopLabels)
                                  .Select(kv => kv.Key)
                                  .ToArray();

            var bigAreas = new List<int>(takeTopLabels * perLabelTop);
            int total = q.Rows * q.Cols;

            foreach (var lab in topLabels)
            {
                using var mask = new Mat();
                Cv2.Compare(q, new Scalar(lab), mask, CmpType.EQ); // 255 onde q==lab

                using var labels = new Mat();
                using var stats = new Mat();
                using var centroids = new Mat();

                // CCWStats em imagem binária (0/255)
                Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8, MatType.CV_32S);

                int n = stats.Rows; // linha 0 = background
                var areas = new List<int>(n);
                for (int i = 1; i < n; i++)
                {
                    int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                    if (area > 0) areas.Add(area);
                }
                areas.Sort(); areas.Reverse();
                bigAreas.AddRange(areas.Take(perLabelTop));

                labels.Dispose(); stats.Dispose(); centroids.Dispose();
            }

            long topArea = bigAreas.OrderByDescending(a => a).Take(takeTopLabels).Sum(); // pega os N maiores globais
            return total > 0 ? (double)topArea / total : 0.0;
        }
    }
}
