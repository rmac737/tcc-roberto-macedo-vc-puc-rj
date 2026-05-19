using ICAONet;
using ICAONet.Background;
using ICAONet.ClosedMouth;
using ICAONet.DTO;
using ICAONet.Emotion;
using ICAONet.Enums;
using ICAONet.EyesOpen;
using ICAONet.FaceCrop;
using ICAONet.FaceDetection;
using ICAONet.FaceMatch;
using ICAONet.GazeEstimation;
using ICAONet.GlassDetection;
using ICAONet.HeadCoverDetection;
using ICAONet.HeadPose;
using ICAONet.ImageQuality;
using ICAONet.Landmarks;
using ICAONet.MakeUpDetection;
using ICAONet.RedEyes;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FACENet
{
    public class FaceNet
    {
        private Utils _utils;
        private FaceMatch _match;
        public FaceNet()
        {
            _utils = new Utils();
            _match = new FaceMatch();
        }

        public async Task<EvaluationFaceMatchResultDTO> Match1x1ImageAsync(byte[] imageProbe, byte[] imageReference, double threshold)
        {
            var result = await _match.Match1x1Async(imageProbe, imageReference, threshold);
            return new EvaluationFaceMatchResultDTO()
            {
                IsMatch = result.Item1,
                Score = result.Item2
            };
        }

        public async Task<EvaluationFaceMatchResultDTO> Match1x1TemplateAsync(float[] imageProbe, float[] imageReference, double threshold)
        {
            var result = await _match.Match1x1Async(imageProbe, imageReference, threshold);
            return new EvaluationFaceMatchResultDTO()
            {
                IsMatch = result.Item1,
                Score = result.Item2
            };
        }

        public async Task<EvaluationFaceMatchResultDTO> Match1x1ImagexTemplateAsync(byte[] imageProbe, float[] imageReference, double threshold)
        {
            var result = await _match.Match1x1Async(imageProbe, imageReference, threshold);
            return new EvaluationFaceMatchResultDTO()
            {
                IsMatch = result.Item1,
                Score = result.Item2
            };
        }
        public async Task<float[]> ExtractTemplateAsync(byte[] imageProbe)
        {
            //var result = await _match.GetEmbedding(null);
            //return result;
            return null;
        }
    }
}
