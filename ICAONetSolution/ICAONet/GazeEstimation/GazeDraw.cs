using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Point = OpenCvSharp.Point;

namespace ICAONet.GazeEstimation
{
    public class GazeDraw
    {
        /// Desenha uma seta 2D indicando a direção do olhar a partir de (cx, cy).
        public static void DrawGazeArrow(
            Mat img, Point origin, float yawDeg, float pitchDeg,
            int length = 80, Scalar? color = null, int thickness = 2)
        {
            color ??= new Scalar(0, 255, 255); // amarelo
                                               // Aproximação em 2D: yaw move horizontal, pitch move vertical (Y cresce p/ baixo)
            double yawRad = yawDeg * Math.PI / 180.0;
            double pitchRad = pitchDeg * Math.PI / 180.0;

            int endX = origin.X + (int)(length * Math.Sin(yawRad));
            int endY = origin.Y - (int)(length * Math.Sin(pitchRad));

            Cv2.ArrowedLine(img, origin, new Point(endX, endY), color.Value, thickness, LineTypes.AntiAlias, 0, 0.25);
        }

        /// Desenha uma “área” (setor/cone) do olhar com transparência.
        /// yaw/pitch definem a direção central; fovYaw/FovPitch definem a abertura em graus.
        public static void DrawGazeCone(
            Mat img, Point origin, float yawDeg, float pitchDeg,
            float fovYawDeg = 15f, float fovPitchDeg = 10f,
            int radius = 100, Scalar? fillBgr = null, double alpha = 0.35, int edgeThickness = 2)
        {
            fillBgr ??= new Scalar(0, 255, 0); // verde

            // Gerar um polígono aproximando o setor elíptico (considera fov diferente em yaw/pitch)
            var poly = BuildGazeSectorPolygon(origin, yawDeg, pitchDeg, fovYawDeg, fovPitchDeg, radius, steps: 24);

            // Overlay com transparência
            using var overlay = img.Clone();
            Cv2.FillConvexPoly(overlay, poly, fillBgr.Value, LineTypes.AntiAlias);
            Cv2.AddWeighted(overlay, alpha, img, 1 - alpha, 0, img);

            // contorno
            Cv2.Polylines(img, new[] { poly }, false, fillBgr.Value, edgeThickness, LineTypes.AntiAlias);
        }

        /// Constrói o polígono do setor (linha central + envelope por fovYaw/fovPitch).
        private static Point[] BuildGazeSectorPolygon(
            Point origin, float yawDeg, float pitchDeg,
            float fovYawDeg, float fovPitchDeg, int radius, int steps = 24)
        {
            // Vetor central (em 2D)
            double yawRad = yawDeg * Math.PI / 180.0;
            double pitchRad = pitchDeg * Math.PI / 180.0;

            // Para gerar o envelope, varremos offsets dentro do FOV (em yaw), e aplicamos pitch junto.
            // Isso vira uma “fatia” suave.
            var pts = new List<Point> { origin };

            for (int i = -steps; i <= steps; i++)
            {
                // t vai de -1..+1
                double t = i / (double)steps;
                double yawOff = t * fovYawDeg * Math.PI / 180.0;
                // pitch alonga/encurta verticalmente
                double pitchOff = t * fovPitchDeg * Math.PI / 180.0;

                // direção atual = central + offset
                double yawTot = yawRad + yawOff;
                double pitchTot = pitchRad + pitchOff;

                int x = origin.X + (int)(radius * Math.Sin(yawTot));
                int y = origin.Y - (int)(radius * Math.Sin(pitchTot));
                pts.Add(new Point(x, y));
            }

            // Fecha retornando à origem
            pts.Add(origin);
            return pts.ToArray();
        }
    }
}
