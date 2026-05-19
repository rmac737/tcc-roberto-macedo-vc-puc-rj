using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet
{
    public class ICAONetParams
    {
        public byte[] Image { get; set; } = Array.Empty<byte>();
        public bool RemoveBackground { get; set; } 
        public int WidthResultImage { get; set; } 
        public int HeightResultImage { get; set; } 
        public int DpiResultImage { get; set; }
    }
}
