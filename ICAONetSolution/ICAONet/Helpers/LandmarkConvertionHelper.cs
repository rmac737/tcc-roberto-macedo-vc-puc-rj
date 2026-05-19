using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.Helpers
{
    public static class LandmarkConvertionHelper
    {
        static int[] Map68to468 = new int[]
        {
            // Contorno do rosto (0–16)
            127, 234, 93, 132, 58, 172, 136, 150, 176,
            152, 400, 377, 378, 365, 397, 288, 361,

            // Sobrancelha direita (17–21)
            70, 63, 105, 66, 107,

            // Sobrancelha esquerda (22–26)
            336, 296, 334, 293, 300,

            // Nariz (27–35)
            168, 6, 197, 195, 5, 4, 1, 98, 327,

            // Olho direito (36–41)
            33, 160, 158, 133, 153, 144,

            // Olho esquerdo (42–47)
            362, 385, 387, 263, 373, 380,

            // Boca externa (48–59)
            61, 146, 91, 181, 84, 17, 314, 405,
            321, 375, 291, 308,

            // Boca interna (60–67)
            78, 95, 88, 178, 87, 14, 317, 402
        };

        public static Point3f[] To68(Point3f[] landmarks468)
        {
            if (landmarks468 == null || landmarks468.Length < 468)
                throw new ArgumentException("Array inválido: deve ter pelo menos 468 pontos.");

            return Map68to468.Select(idx => landmarks468[idx]).ToArray();
        }
    }
}
