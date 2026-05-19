using ICAONet.ImageQuality;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.DTO
{
    public class EvaluationFaceMatchResultDTO
    {
        public bool IsMatch { get; set; }
        public double Score { get; set; }
    }
}
