using Microsoft.Extensions.ObjectPool;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.FaceDetection
{
    public sealed class OpenCvNetPooledFaceMatchPolicy : PooledObjectPolicy<Net>
    {
        private readonly string _protoPath, _modelPath;

        public OpenCvNetPooledFaceMatchPolicy(string protoPath, string modelPath)
        {
            _protoPath = protoPath;
            _modelPath = modelPath;
        }

        public override Net Create() => CvDnn.ReadNetFromCaffe(_protoPath, _modelPath)!;

        public override bool Return(Net obj)
        {
            return true;
        }
    }
}
