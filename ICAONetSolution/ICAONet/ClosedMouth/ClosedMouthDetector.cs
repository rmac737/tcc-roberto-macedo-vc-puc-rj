using ICAONet.Enums;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.ClosedMouth
{
    public class ClosedMouthDetector
    {
        public enum MouthState { Closed, SlightlyOpen, Open }

        // Indices FaceMesh (468 pts)
        private const int UpperLipCenter = 13;
        private const int LowerLipCenter = 14;
        private const int MouthLeftCorner = 61;   // canto esquerdo (do ponto de vista do observador)
        private const int MouthRightCorner = 291; // canto direito

        public async Task<(bool,string)>
            Evaluate(Point2f[] lm, double thClosed = 0.20, double thOpen = 0.35)
        {

            return await Task.Run(() =>
            {
                if (lm == null || lm.Length < 292)
                    throw new ArgumentException("Precisam ser 468 pontos do FaceMesh.");

                Point2f pUp = lm[UpperLipCenter];
                Point2f pLo = lm[LowerLipCenter];
                Point2f pL = lm[MouthLeftCorner];
                Point2f pR = lm[MouthRightCorner];

                double open = Dist(pUp, pLo);
                double width = Dist(pL, pR);
                if (width <= 1e-6) width = 1e-6; // evita divisão por zero

                double ratio = open / width;

                MouthState state =
                    (ratio < thClosed) ? MouthState.Closed :
                    (ratio >= thOpen) ? MouthState.Open : MouthState.SlightlyOpen;

                #region Implementação da Regra"

                if(state == MouthState.Closed)
                {
                    return (true, $"{EnumICAOResult.PASSED}.");
                }
                else
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED}.");
                }

                #endregion
                //return (state, ratio, open, width);
            });
        }

        private double Dist(Point2f a, Point2f b)
        {
            double dx = a.X - b.X;
            double dy = a.Y - b.Y;
            return Math.Sqrt(dx * dx + dy * dy);
        }
    }
}
