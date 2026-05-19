using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.Enums
{
    public enum OutputMode
    {
        PixelsAbsolutos,      // já em coordenadas da imagem
        NormalizadoZeroUm,    // [0,1] relativo ao crop
        NormalizadoMenosUmUm  // [-1,1] relativo ao crop
    }
}
