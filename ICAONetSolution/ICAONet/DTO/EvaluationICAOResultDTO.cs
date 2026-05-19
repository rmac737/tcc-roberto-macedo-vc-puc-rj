using ICAONet.ImageQuality;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.DTO
{
    public class EvaluationICAOResultDTO
    {
        public byte[] OriginalImage { get; set; } = Array.Empty<byte>();
        public byte[] ICAOImage { get; set; } = Array.Empty<byte>();
        public byte[] ICAOImageWithoutBackground { get; set; } = Array.Empty<byte>();
        public (bool, string) IsImageOkToBeProcessed { get; set; }
        public (bool, string)  IsImagePixelatedComplicance { get; set; }
        public (bool, string) IsImagePosterizedCompliance { get; set; }
        public (bool, string) IsFocusCompliance { get; set; }
        public (bool, string) IsMouthClosed { get; set; }
        public (bool, string) IsEyesOpened { get; set; }
        public (bool, string) IsGlassCompliance { get; set; }
        public (bool, string) IsSmilingCompliance { get; set; }
        public (bool, string) IsEyeGazeCompliance { get; set; }
        public (bool, string) IsHeadCoverCompliance { get; set; }
        public (bool, string) IsRedEyeCompliance { get; set; }
        public (bool, string) IsMakeUpCompliance { get; set; }
        public (bool, string) IsHeadPoseCompliance { get; set; }
        public (bool, string) IsLightningAsymmetryCompliance { get; set; }
        public (bool, string) IsLightReflextionCompliance { get; set; }
        public (bool, string) IsShadowCompliance { get; set; }
        public (bool, string) IsColorUniformityCompliance { get; set; }
    }
}
