using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using ICAONet.Enums;
using ICAONet.Helpers;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.ImageQuality
{
    public class QualityChecker
    {
        YoloPredictor? yoloPredictorPixelated;
        YoloPredictor? yoloPredictorFocused;
        private Utils _utils;

        public QualityChecker()
        {
            _utils = new Utils();
            yoloPredictorPixelated = new YoloPredictor(_utils.LoadResource("ICAONet.ImageQuality.pixeled_no_pixeled.onnx"));
            yoloPredictorFocused = new YoloPredictor(_utils.LoadResource("ICAONet.ImageQuality.focused_no_focused.onnx"));
        }
        public async Task<(bool,string)> IsImagePixelated(Mat imagem)
        {
            // Run model
            YoloResult<Classification> result = await yoloPredictorPixelated!.ClassifyAsync(imagem.ToBytes());

            foreach (var pred in result.ToList())
            {
                if (pred.Confidence >= 0.50)
                {
                    if (pred.Name.Name.Contains("no"))
                    {
                        return (true, $"{EnumICAOResult.PASSED}. Image is not pixelated.");
                    }
                    else
                    {
                        return (false, $"{EnumICAOResult.NOT_PASSED}. Image is pixelated.");
                    }
                }
            }

            return (false, $"{EnumICAOResult.NOT_PASSED}.");
        }        
        public async Task<(bool,string)> IsImageOutOfFocus(Mat imagem)
        {
            YoloResult<Classification> result = await yoloPredictorFocused!.ClassifyAsync(imagem.ToBytes());

            foreach (var pred in result.ToList())
            {
                if (pred.Confidence >= 0.50)
                {
                    if (pred.Name.Name.Contains("no"))
                    {
                        return (false, $"{EnumICAOResult.NOT_PASSED}. Image is not focused.");
                    }
                    else
                    {
                        return (true, $"{EnumICAOResult.PASSED}. Image is focused.");
                    }
                }
            }

            return (false, $"{EnumICAOResult.NOT_PASSED}.");
        }
        public async Task<(bool,string)> IsImagePosterized(Mat imagem)
        {

            return await Task.Run(() =>
            {
                var (posterized, reason) = PosterizationHelper.Check(imagem.ToBytes());

                var (posterized2, reason2) = PosterizationV2Helper.Check(imagem.ToBytes());

                string outText = $"{reason} - {reason2}";

                if (posterized && posterized2)
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED}. {outText}.");
                }
                else
                {
                    return (true, $"{EnumICAOResult.PASSED}. {outText}.");
                }
            });
        }

        public async Task<(bool, string)> IsImageUniformLightning(Mat imagem, Point2f[] points)
        {
            return await Task.Run(() =>
            {
                var result = LightingHomogeneityEvaluator.Evaluate(imagem, points);

                if(result.Passed)
                {
                    return (true, result.Reason);
                }
                else
                {
                    return (false, result.Reason);
                } 
            });
        }

        public async Task<(bool,string)> IsImageOkToBeProcessed(byte[] image)
        {
            return await Task.Run(() =>
            {
                using Mat imagem = Cv2.ImDecode(image, ImreadModes.AnyColor);

                if (imagem.Empty())
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED.ToString()}; The image cannot be null or empty.");
                }

                if (imagem.Width < 480 || imagem.Height < 640)
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED.ToString()}; The image has low resolution. The minimum required is 480x640.");
                }

                byte[] header = image.Take(8).ToArray();

                if (!((header[0] == 0xFF && header[1] == 0xD8) || (header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47))) //JPEG - PNG
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED.ToString()}; Only JPG/JPEG or PNG format allowed.");
                }

                return (true, EnumICAOResult.PASSED.ToString());
            });
        }
    }
}
