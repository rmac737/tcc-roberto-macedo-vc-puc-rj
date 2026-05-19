using ICAONet.Enums;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.EyesOpen
{
    public class EyesOpenEstimator
    {
        // Índices do MediaPipe FaceMesh (468 pts)
        private readonly int[] LeftEye = { 33, 160, 158, 133, 153, 144 }; // p1..p6
        private readonly int[] RightEye = { 362, 385, 387, 263, 373, 380 }; // p1..p6

        /// <summary>
        /// Retorna (earEsq, earDir, earMedio) e flags (esqAberto, dirAberto)
        /// landmarks: array de 468 pontos (x,y,z). x,y podem estar normalizados (0..1) ou em px.
        /// imgWidth/Height: use o tamanho da imagem; se landmarks já estão em px, passe os mesmos tamanhos.
        /// </summary>
        public async Task<(bool, string)>
            Compute(Point3f[] landmarks, int imgWidth, int imgHeight,
                    float openThresh = 0.27f, float closeThresh = 0.22f)
        {
            return await Task.Run(() =>
            {
                if (landmarks == null || landmarks.Length < 388)
                    return (false, $"{EnumICAOResult.NOT_PASSED.ToString()}. Esperados 468 pontos do FaceMesh.");

                float earL = Ear(landmarks, LeftEye, imgWidth, imgHeight);
                float earR = Ear(landmarks, RightEye, imgWidth, imgHeight);
                float earAvg = (earL + earR) * 0.5f;

                // Histerese simples para estabilidade: se quiser, persista estado entre frames
                bool leftOpen = earL >= openThresh;
                bool rightOpen = earR >= openThresh;

                if (leftOpen && rightOpen)
                {
                    return (true, $"{EnumICAOResult.PASSED}.");
                }
                else
                {
                    return (false, $"Threshold: {openThresh} -> Olho Esquerdo: {(leftOpen ? EnumICAOResult.PASSED.ToString() : EnumICAOResult.NOT_PASSED.ToString())}. Threshold calculado: {earL}. Olho Direito: {(rightOpen ? EnumICAOResult.PASSED.ToString() : EnumICAOResult.NOT_PASSED.ToString())}. Threshold calculado: {earR}");
                }
                //return (earL, earR, earAvg, leftOpen, rightOpen);
            });
        }

        private float Ear(Point3f[] lm, int[] idx, int w, int h)
        {
            var p1 = ToPx(lm[idx[0]], w, h);
            var p2 = ToPx(lm[idx[1]], w, h);
            var p3 = ToPx(lm[idx[2]], w, h);
            var p4 = ToPx(lm[idx[3]], w, h);
            var p5 = ToPx(lm[idx[4]], w, h);
            var p6 = ToPx(lm[idx[5]], w, h);

            float d26 = Dist(p2, p6);
            float d35 = Dist(p3, p5);
            float d14 = Dist(p1, p4);

            if (d14 <= 1e-6f) return 0f; // evita div/0
            return (d26 + d35) / (2f * d14);
        }

        private (float X, float Y) ToPx(Point3f p, int w, int h)
            => (p.X * w, p.Y * h); // se já estiver em px, troque por: (p.X, p.Y)

        private float Dist((float X, float Y) a, (float X, float Y) b)
        {
            float dx = a.X - b.X, dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

    }
}
