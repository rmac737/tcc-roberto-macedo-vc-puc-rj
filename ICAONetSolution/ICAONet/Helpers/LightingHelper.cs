using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.Helpers
{
    public record LightingParams(
        double UniformityMax, double LrDiffMax,
        double ShadowMaxPct, double SpecularMaxPct,
        double ShadowK, byte SpecularVMin, byte SpecularSMax,
        int WorkSize
    )
    {
        public static LightingParams Default => new(
            UniformityMax: 0.70,   // std/mean do low-pass L*
            LrDiffMax: 0.40,   // diferença relativa (esq×dir)
            ShadowMaxPct: 4.0,    // % área com sombra/penumbra
            SpecularMaxPct: 0.6,    // % área com highlights/reflexos
            ShadowK: 1.0,    // L < (μ_local - K·σ_local)
            SpecularVMin: 235,    // HSV V alto
            SpecularSMax: 40,     // HSV S baixo (branco brilhoso)
            WorkSize: 256
        );
    }

    public record LightingReport(
        bool Passed,
        double UniformityRatio,
        double LRMeanDiff,
        double ShadowAreaPct,
        double SpecularAreaPct,
        string Reason
    );
    public static class LightingHomogeneityEvaluator
    {
        // --- Índices do FaceMesh para olhos e boca (MediaPipe) ---
        static readonly int[] LeftEyeIdx = { 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 };
        static readonly int[] RightEyeIdx = { 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466 };
        static readonly int[] MouthIdx = { 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87 };

        /// <summary>
        /// Avalia iluminação homogênea (ICAO-like) usando somente C# + OpenCvSharp.
        /// Entrada: Mat BGR completo e 468 landmarks (FaceMesh) no mesmo espaço da imagem.
        /// </summary>
        public static LightingReport Evaluate(Mat bgr, Point2f[] landmarks468, LightingParams p = null)
        {
            if (bgr.Empty()) throw new ArgumentException("Imagem vazia.");
            if (landmarks468 is null || landmarks468.Length < 468)
                throw new ArgumentException("São necessários 468 pontos do FaceMesh.");

            p ??= LightingParams.Default;

            // 1) FaceRect a partir dos landmarks (+ leve expansão)
            var faceRect = InflateClamp(BoundingRect(landmarks468), bgr.Size(), 0.10, 0.12);

            // 2) Construção da máscara de pele (tamanho da imagem): elipse interna - olhos - boca
            using var skinMaskFull = BuildSkinMask(bgr.Size(), landmarks468, faceRect,
                                                   ellipseScaleX: 0.92, ellipseScaleY: 0.97,
                                                   foreheadBiasFrac: -0.04,
                                                   eyeInflate: 1.45, mouthInflate: 1.25,
                                                   smoothKernel: 7);

            // 3) ROI para processamento (apenas área da face)
            var roiRect = InflateClamp(faceRect, bgr.Size(), 0.08, 0.08);
            using var roiBgr = new Mat(bgr, roiRect);
            using var roiMask = new Mat(skinMaskFull, roiRect);

            // Se a máscara dentro da ROI for muito pequena, falha por falta de área útil
            int maskArea = Cv2.CountNonZero(roiMask);
            if (maskArea < 64) // ~ área mínima simbólica
                return new LightingReport(false, 1, 1, 100, 100, "Área de pele insuficiente na ROI.");

            // 4) Redimensiona para processamento estável
            using var resized = roiBgr.Resize(new OpenCvSharp.Size(p.WorkSize, p.WorkSize), 0, 0, InterpolationFlags.Area);
            using var maskSmall = roiMask.Resize(new OpenCvSharp.Size(p.WorkSize, p.WorkSize), 0, 0, InterpolationFlags.Nearest);

            // 5) Espaços de cor
            using var lab = new Mat(); Cv2.CvtColor(resized, lab, ColorConversionCodes.BGR2Lab);
            var labSplit = Cv2.Split(lab);
            using var L = labSplit[0]; // L* (0..255 aprox)

            using var hsv = new Mat(); Cv2.CvtColor(resized, hsv, ColorConversionCodes.BGR2HSV);
            var hsvSplit = Cv2.Split(hsv);
            using var H = hsvSplit[0];
            using var S = hsvSplit[1];
            using var V = hsvSplit[2];

            // 6) Low-pass do L* para medir uniformidade e assimetria
            using var Lf = new Mat(); L.ConvertTo(Lf, MatType.CV_32F, 1.0 / 255.0);
            int k = NextOdd(Math.Max(5, (int)(p.WorkSize * 0.125))); // ~32 p/ 256
            using var illum = new Mat();
            Cv2.GaussianBlur(Lf, illum, new OpenCvSharp.Size(k, k), 0, 0, BorderTypes.Reflect101);

            // Uniformidade: std/mean do low-pass de L* sob a máscara
            var (meanIllum, stdIllum) = MeanStd(illum, maskSmall);
            double uniformity = (meanIllum > 1e-6) ? (stdIllum / meanIllum) : 1.0;

            // 7) Sombras/penumbras: Lf < μ_local - K·σ_local, restrito à máscara
            using var localMean = new Mat();
            using var localSq = new Mat();
            Cv2.GaussianBlur(Lf, localMean, new OpenCvSharp.Size(k, k), 0, 0, BorderTypes.Reflect101);
            Cv2.GaussianBlur(Lf.Mul(Lf), localSq, new OpenCvSharp.Size(k, k), 0, 0, BorderTypes.Reflect101);
            using var localVar = localSq - localMean.Mul(localMean);
            Cv2.Max(localVar, 0, localVar);
            using var localStd = new Mat(); Cv2.Sqrt(localVar, localStd);

            using var thresh = new Mat(); Cv2.Subtract(localMean, localStd * p.ShadowK, thresh);
            using var maskShadowLT = new Mat(); Cv2.Compare(Lf, thresh, maskShadowLT, CmpType.LT); // Lf < thresh
            using var sLow = new Mat(); Cv2.Compare(S, p.SpecularSMax, sLow, CmpType.LE); // evita cromas fortes
            using var shadowMask = new Mat();
            Cv2.BitwiseAnd(maskShadowLT, sLow, shadowMask);
            Cv2.BitwiseAnd(shadowMask, maskSmall, shadowMask);
            MorphClean(shadowMask, 3);
            double shadowPct = 100.0 * Cv2.CountNonZero(shadowMask) / Math.Max(1, Cv2.CountNonZero(maskSmall));

            // 8) Reflexos (specular): V muito alto & S baixo + estouro de canal
            int vThr = Math.Max(p.SpecularVMin, PercentileHigh(V, maskSmall, 98));

            // V ≥ vThr  e  S ≤ SpecularSMax
            using var vHi = new Mat(); Cv2.Compare(V, vThr, vHi, CmpType.GE);
            using var sLow2 = new Mat(); Cv2.Compare(S, p.SpecularSMax, sLow2, CmpType.LE);
            using var spec1 = new Mat(); Cv2.BitwiseAnd(vHi, sLow2, spec1);

            // (opcional) Estouro de canal como pista adicional
            using var satOverflow = SaturationOverflowMask(resized, 250);
            Cv2.BitwiseOr(spec1, satOverflow, spec1);

            // 8.2) Requerer "brilho com borda": contraste local alto no L* (std local acima de um limiar)
            using var localStdBin = new Mat();
            Cv2.Threshold(localStd, localStdBin, 0.05, 255, ThresholdTypes.Binary); // Lf é 0..1; 0.05 ~ 5% var.
            localStdBin.ConvertTo(localStdBin, MatType.CV_8U, 255.0); // binário 0/255

            using var spec2 = new Mat();
            Cv2.BitwiseAnd(spec1, localStdBin, spec2);

            // 8.3) Limpeza morfológica + filtro por área relativa (remove ruído e zonas gigantes)
            Cv2.BitwiseAnd(spec2, maskSmall, spec2);
            MorphClean(spec2, 3);
            CleanByArea(spec2, minFrac: 0.0002, maxFrac: 0.02, withinMask: maskSmall); // 0.02 = 2% da área da máscara

            double specPct = 100.0 * Cv2.CountNonZero(spec2) / Math.Max(1, Cv2.CountNonZero(maskSmall));

            // 9) Assimetria esquerda × direita (no low-pass, sob máscara)
            double lrDiff = LeftRightMeanDiff(illum, maskSmall);

            // 10) Decisão
            var fails = new List<string>();
            if (uniformity > p.UniformityMax) fails.Add($"Uniformidade ruim (std/mean={uniformity:0.000} > {p.UniformityMax:0.00});");
            if (lrDiff > p.LrDiffMax) fails.Add($"Assimetria esq×dir (Δ={lrDiff:0.000} > {p.LrDiffMax:0.00});");
            if (shadowPct > p.ShadowMaxPct) fails.Add($"Sombras {shadowPct:0.1}% > {p.ShadowMaxPct:0.1}% ;");
            if (specPct > p.SpecularMaxPct) fails.Add($"Reflexos {specPct:0.2}% > {p.SpecularMaxPct:0.2}% ;");

            bool passed = fails.Count == 0;
            string reason = passed ? "OK: iluminação homogênea nas regiões de pele." : string.Join("; ", fails);

            // cleanup splits
            foreach (var m in labSplit) m.Dispose();
            foreach (var m in hsvSplit) m.Dispose();
            
            return new LightingReport(passed, uniformity, lrDiff, shadowPct, specPct, reason);
        }

        // ----------------- Helpers principais -----------------
        private static int PercentileHigh(Mat channel8u, Mat mask8u, int percentile /* ex.: 98 */)
        {
            // Histograma 0..255 sob a máscara
            var hist = new Mat();
            Cv2.CalcHist(
                new[] { channel8u },
                new[] { 0 },
                mask8u,
                hist,
                1,
                new[] { 256 },
                new Rangef[] { new Rangef(0, 256) } // <-- precisa ser array
            );

            // total de pixels válidos
            double total = Cv2.CountNonZero(mask8u);
            if (total <= 0) return 255;

            // acumulado de cima para baixo (percentil alto)
            double target = total * (100.0 - percentile) / 100.0;
            double acc = 0;
            for (int v = 255; v >= 0; v--)
            {
                acc += hist.Get<float>(v);
                if (acc >= target) return v;
            }
            return 255;
        }

        private static void CleanByArea(Mat binMask, double minFrac, double maxFrac, Mat withinMask)
        {
            int total = Math.Max(1, Cv2.CountNonZero(withinMask));
            int minArea = Math.Max(1, (int)Math.Round(total * minFrac));
            int maxArea = Math.Max(minArea + 1, (int)Math.Round(total * maxFrac));

            using var labels = new Mat();
            using var stats = new Mat();
            using var cents = new Mat();

            // binMask deve ser 8U 0/255
            var tmp = binMask.Threshold(1, 255, ThresholdTypes.Binary);
            int n = Cv2.ConnectedComponentsWithStats(tmp, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);

            // zera fora do range de área
            var data = binMask.Clone();
            data.SetTo(Scalar.Black);

            for (int i = 1; i < n; i++)
            {
                int area = stats.Get<int>(i, (int)ConnectedComponentsTypes.Area);
                if (area >= minArea && area <= maxArea)
                {
                    using var comp = new Mat();
                    Cv2.Compare(labels, i, comp, CmpType.EQ);  // <-- substitui labels == i
                    Cv2.BitwiseOr(data, comp, data);
                }
            }

            data.ConvertTo(binMask, binMask.Type());
        }
        private static Rect BoundingRect(Point2f[] pts)
        {
            int minX = int.MaxValue, minY = int.MaxValue, maxX = int.MinValue, maxY = int.MinValue;
            foreach (var p in pts)
            {
                if (p.X < minX) minX = (int)Math.Floor(p.X);
                if (p.Y < minY) minY = (int)Math.Floor(p.Y);
                if (p.X > maxX) maxX = (int)Math.Ceiling(p.X);
                if (p.Y > maxY) maxY = (int)Math.Ceiling(p.Y);
            }
            return new Rect(minX, minY, Math.Max(1, maxX - minX + 1), Math.Max(1, maxY - minY + 1));
        }

        private static Rect InflateClamp(Rect r, OpenCvSharp.Size bounds, double scaleX, double scaleY)
        {
            int cx = r.X + r.Width / 2;
            int cy = r.Y + r.Height / 2;
            int w = (int)Math.Round(r.Width * (1.0 + scaleX));
            int h = (int)Math.Round(r.Height * (1.0 + scaleY));
            var rr = new Rect(cx - w / 2, cy - h / 2, w, h);
            rr.X = Math.Max(0, rr.X);
            rr.Y = Math.Max(0, rr.Y);
            rr.Width = Math.Min(bounds.Width - rr.X, rr.Width);
            rr.Height = Math.Min(bounds.Height - rr.Y, rr.Height);
            if (rr.Width < 1 || rr.Height < 1) rr = new Rect(r.X, r.Y, Math.Max(1, r.Width), Math.Max(1, r.Height));
            return rr;
        }

        /// <summary>
        /// Elipse interna no faceRect - olhos - boca, suavizada. Retorna máscara 8U (0/255) no tamanho da imagem.
        /// </summary>
        private static Mat BuildSkinMask(OpenCvSharp.Size imgSize, Point2f[] lm, Rect faceRect,
                                         double ellipseScaleX, double ellipseScaleY,
                                         double foreheadBiasFrac, double eyeInflate, double mouthInflate, int smoothKernel)
        {
            var mask = new Mat(imgSize, MatType.CV_8U, Scalar.Black);

            // Elipse principal (evita cabelo/orelhas)
            double cx = faceRect.X + faceRect.Width * 0.5;
            double cy = faceRect.Y + faceRect.Height * (0.5 + foreheadBiasFrac);
            var axes = new OpenCvSharp.Size(faceRect.Width * 0.5 * ellipseScaleX,
                                faceRect.Height * 0.5 * ellipseScaleY);
            Cv2.Ellipse(mask, new OpenCvSharp.Point((int)cx, (int)cy), axes, 0, 0, 360, Scalar.White, -1, LineTypes.AntiAlias);

            // "Buracos" para olhos e boca
            CutEllipse(mask, lm, LeftEyeIdx, eyeInflate);
            CutEllipse(mask, lm, RightEyeIdx, eyeInflate);
            CutEllipse(mask, lm, MouthIdx, mouthInflate);

            if (smoothKernel > 1)
            {
                using var ker = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(smoothKernel, smoothKernel));
                Cv2.MorphologyEx(mask, mask, MorphTypes.Open, ker);
                Cv2.MorphologyEx(mask, mask, MorphTypes.Close, ker);
            }

            return mask;
        }

        private static void CutEllipse(Mat mask, Point2f[] lm, int[] idx, double inflate)
        {
            var pts = idx.Select(i => lm[i]).ToArray();
            var r = Cv2.BoundingRect(pts.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray());
            double cx = r.X + r.Width * 0.5;
            double cy = r.Y + r.Height * 0.5;
            var axes = new OpenCvSharp.Size(Math.Max(2, r.Width * 0.5 * inflate),
                                Math.Max(2, r.Height * 0.5 * inflate));
            Cv2.Ellipse(mask, new OpenCvSharp.Point((int)cx, (int)cy), axes, 0, 0, 360, Scalar.Black, -1, LineTypes.AntiAlias);
        }

        private static (double mean, double std) MeanStd(Mat f32, Mat mask8u)
        {
            Cv2.MeanStdDev(f32, out Scalar meanS, out Scalar stdS, mask8u);
            return (meanS.Val0, stdS.Val0);
        }

        private static int NextOdd(int v) => (v % 2 == 1) ? v : (v + 1);

        private static void MorphClean(Mat mask, int k)
        {
            using var ker = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(k, k));
            Cv2.MorphologyEx(mask, mask, MorphTypes.Open, ker);
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, ker);
        }

        private static Mat SaturationOverflowMask(Mat bgr, byte satThr)
        {
            var ch = Cv2.Split(bgr); // B,G,R (8U)
            using var maxCh = new Mat(); using var minCh = new Mat();
            Cv2.Max(ch[0], ch[1], maxCh); Cv2.Max(maxCh, ch[2], maxCh);
            Cv2.Min(ch[0], ch[1], minCh); Cv2.Min(minCh, ch[2], minCh);

            using var diff = maxCh - minCh;     // quão colorido
            using var vMask = new Mat(); Cv2.Threshold(maxCh, vMask, satThr, 255, ThresholdTypes.Binary);
            using var grayish = new Mat(); Cv2.Threshold(diff, grayish, 12, 255, ThresholdTypes.BinaryInv);

            var spec = new Mat();
            Cv2.BitwiseAnd(vMask, grayish, spec);

            foreach (var c in ch) c.Dispose();
            return spec;
        }

        private static double LeftRightMeanDiff(Mat lowpassF32, Mat mask8u)
        {
            int mid = lowpassF32.Cols / 2;
            using var leftMask = new Mat(mask8u.Size(), MatType.CV_8U, Scalar.Black);
            using var rightMask = new Mat(mask8u.Size(), MatType.CV_8U, Scalar.Black);
            Cv2.Rectangle(leftMask, new Rect(0, 0, mid, lowpassF32.Rows), Scalar.White, -1);
            Cv2.Rectangle(rightMask, new Rect(mid, 0, lowpassF32.Cols - mid, lowpassF32.Rows), Scalar.White, -1);
            Cv2.BitwiseAnd(leftMask, mask8u, leftMask);
            Cv2.BitwiseAnd(rightMask, mask8u, rightMask);

            var (ml, _) = MeanStd(lowpassF32, leftMask);
            var (mr, _) = MeanStd(lowpassF32, rightMask);
            double m = Math.Max(1e-6, (ml + mr) * 0.5);
            return Math.Abs(ml - mr) / m;
        }
    }
}
