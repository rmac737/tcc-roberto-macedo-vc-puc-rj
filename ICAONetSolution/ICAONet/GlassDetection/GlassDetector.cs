using Compunet.YoloSharp;
using ICAONet.Enums;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

namespace ICAONet.GlassDetection
{
    public class GlassDetector
    {
        YoloPredictor? yoloPredictor;
        private Utils _utils;
        public GlassDetector()
        {
            _utils = new Utils();
            yoloPredictor = new YoloPredictor(_utils.LoadResource("ICAONet.GlassDetection.glass_no_glass.onnx"));
        }

        public async Task<(bool,string)> Predict(Mat imagem)
        {
            // Run model
            YoloResult<Classification> result = await yoloPredictor!.ClassifyAsync(imagem.ToBytes());

            foreach (var pred in result.ToList())
            {
                if(pred.Confidence >= 0.60)
                {
                    if (pred.Name.Name.Contains("no"))
                    {
                        return (true, $"{EnumICAOResult.PASSED}. Óculos não detectado.");
                    }
                    else
                    {
                        return (false, $"{EnumICAOResult.NOT_PASSED}. Óculos detectado.");
                    }
                }
            }

            return (false, $"{EnumICAOResult.NOT_PASSED}.");
        }
    }
}
