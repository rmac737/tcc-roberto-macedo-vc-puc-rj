
using Compunet.YoloSharp;
using Compunet.YoloSharp.Plotting;
using ICAONet.Enums;
using OpenCvSharp;

namespace ICAONet.HeadCoverDetection
{
    public class HeadCoverDetector
    {
        YoloPredictor? yoloPredictor;
        private Utils _utils;
        public HeadCoverDetector()
        {
            _utils = new Utils();
            yoloPredictor = new YoloPredictor(_utils.LoadResource("ICAONet.HeadCoverDetection.head_cover.onnx"));
        }
        public async Task<(bool, string)> Predict(Mat imagem)
        {
            // Run model
            YoloResult<Classification> result = await yoloPredictor!.ClassifyAsync(imagem.ToBytes());

            foreach (var pred in result.ToList())
            {
                if (pred.Confidence >= 0.50)
                {
                    if (pred.Name.Name.Contains("no"))
                    {
                        return (true, $"{EnumICAOResult.PASSED}. Sem oclusão na cabeça.");
                    }
                    else
                    {
                        return (false, $"{EnumICAOResult.NOT_PASSED}. Oclusão na cabeça detectado.");
                    }
                }
            }

            return (false, $"{EnumICAOResult.NOT_PASSED}.");
        }
    }
}
