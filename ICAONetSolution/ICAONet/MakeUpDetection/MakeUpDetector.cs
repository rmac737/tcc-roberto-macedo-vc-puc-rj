using Compunet.YoloSharp;
using ICAONet.Enums;
using OpenCvSharp;
using System;
using System.Linq;

namespace ICAONet.MakeUpDetection
{
    public class MakeUpDetector
    {
        YoloPredictor? yoloPredictor;
        private Utils _utils;
        public MakeUpDetector()
        {
            _utils = new Utils();
            yoloPredictor = new YoloPredictor(_utils.LoadResource("ICAONet.MakeUpDetection.make_up.onnx"));
        }

        public async Task<(bool, string)> Predict(Mat imagem)
        {
            // Run model
            YoloResult<Detection> result = await yoloPredictor!.DetectAsync(imagem.ToBytes());

            foreach (var item in result.ToList())
            {
                if(item.Confidence > 0.6 && item.Name.Name.Contains("no"))
                {
                    return (true, $"{EnumICAOResult.PASSED}. MakeUp OK.");
                }
                else
                {
                    return (true, $"{EnumICAOResult.NOT_PASSED}. MakeUp not OK. Detected: {item.Name.Name}");
                }
            }
            
            return (false, $"{EnumICAOResult.NOT_PASSED}");
        }
    }
}
