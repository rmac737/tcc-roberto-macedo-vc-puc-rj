using ICAONet.FaceDetection;
using Microsoft.Extensions.ObjectPool;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using SixLabors.ImageSharp;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace ICAONet.FaceMatch
{
    public class FaceMatch
    {
        private readonly InferenceSession _session;
        private Utils _utils;
        private CaffeModel _faceDetection;
        public FaceMatch()
        {
            _utils = new Utils();
            var so = new SessionOptions();
            so.IntraOpNumThreads = Environment.ProcessorCount;
            so.IntraOpNumThreads = 1;
            so.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            so.AppendExecutionProvider_DML();
            so.EnableCpuMemArena = true;

            _session = new InferenceSession(_utils.LoadResource("ICAONet.FaceMatch.glintr100.onnx"), so);
            _faceDetection = new CaffeModel();
        }

        public async Task<string> GetTemplateStringBase64(byte[] image)
        {
            string ret = string.Empty;

            ret = await Task.Run(() =>
            {
                using var bmp = SKBitmap.Decode(_faceDetection.GetFaceCropped(image).Item1);

                if (bmp == null)
                    throw new InvalidOperationException("Não foi possível decodificar a imagem.");

                float[] probe = GetEmbedding(bmp);
                return new Utils().SerializeFloatArrayToStringBase64(probe);
            });

            return ret;
        }
        public async Task<float[]> GetTemplate(byte[] image)
        {
            float[] ret = Array.Empty<float>();

            ret = await Task.Run(() =>
            {
                using var bmp = SKBitmap.Decode(_faceDetection.GetFaceCropped(image).Item1);
                if (bmp == null)
                    throw new InvalidOperationException("Não foi possível decodificar a imagem.");

                float[] probe = GetEmbedding(bmp);
                return probe;
            });

            return ret;
        }
        public async Task<(bool, double)> Match1x1Async(byte[] imageProbe, float[] templateReference, double threshold)
        {
            (bool, double) ret = (false, 0.0);

            ret = await Task.Run(() =>
            {
                using var bmp = SKBitmap.Decode(_faceDetection.GetFaceCropped(imageProbe).Item1);
                if (bmp == null)
                    throw new InvalidOperationException("Não foi possível decodificar a imagem.");

                float[] probe = GetEmbedding(bmp);
                float result = CosineSimilarity(probe, templateReference);

                if (result >= threshold)
                {
                    return (true, result);
                }
                else
                {
                    return (false, result);
                }
            });

            return ret;
        }
        public async Task<(bool, double)> Match1x1Async(byte[] imageProbe, byte[] imageReference, double threshold)
        {
            (bool, double) ret = (false, 0.0);

            ret = await Task.Run(() =>
            {
                using var bmp = SKBitmap.Decode(_faceDetection.GetFaceCropped(imageProbe).Item1);
                if (bmp == null)
                    throw new InvalidOperationException("Não foi possível decodificar a imagem.");

                using var bmp2 = SKBitmap.Decode(_faceDetection.GetFaceCropped(imageReference).Item1);
                if (bmp2 == null)
                    throw new InvalidOperationException("Não foi possível decodificar a imagem.");

                float[] probe = GetEmbedding(bmp);
                float[] reference = GetEmbedding(bmp2);
                float result = CosineSimilarity(probe, reference);

                if (result >= threshold)
                {
                    return (true, result);
                }
                else
                {
                    return (false, result);
                }
            });

            return ret;
        }
        public async Task<(bool, double)> Match1x1Async(float[] templateProbe, float[] templateReference, double threshold)
        {
            (bool, double) ret = (false, 0.0);

            ret = await Task.Run(() =>
            {
                float result = CosineSimilarity(templateProbe, templateReference);

                if (result >= threshold)
                {
                    return (true, result);
                }
                else
                {
                    return (false, result);
                }
            });

            return ret;
        }

        #region "private methods"

        public float[] GetEmbedding(SKBitmap bmp)
        {
            var input = PreprocessImage(bmp);

            var inputs = new[]
            {
            NamedOnnxValue.CreateFromTensor("input.1", input)
        };

            var watch = System.Diagnostics.Stopwatch.StartNew();
            using var results = _session.Run(inputs);
            watch.Stop();
            //Console.WriteLine($"Session Run: {watch.ElapsedMilliseconds}");

            // Evita First + Enumerable
            var tensor = results.First().AsTensor<float>();
            return tensor.ToArray();
        }
        private DenseTensor<float> PreprocessImage(SKBitmap image)
        {
            using var resized = image.Resize(
                new SKImageInfo(112, 112, SKColorType.Rgba8888), SKSamplingOptions.Default);

            if (resized == null)
                throw new InvalidOperationException("Falha ao redimensionar a imagem.");

            var tensor = new DenseTensor<float>(new[] { 1, 3, 112, 112 });

            var span = resized.GetPixelSpan();
            var buffer = tensor.Buffer.Span;

            int hw = 112 * 112;
            int pixelIndex = 0;

            for (int i = 0; i < hw; i++)
            {
                float r = span[pixelIndex + 0];
                float g = span[pixelIndex + 1];
                float b = span[pixelIndex + 2];

                buffer[i] = (r - 127.5f) / 128f;           // R
                buffer[i + hw] = (g - 127.5f) / 128f;      // G
                buffer[i + hw * 2] = (b - 127.5f) / 128f;  // B

                pixelIndex += 4;
            }

            return tensor;
        }
        private float CosineSimilarity(float[] v1, float[] v2)
        {
            float dot = 0f;
            float norm1 = 0f;
            float norm2 = 0f;

            for (int i = 0; i < v1.Length; i++)
            {
                float a = v1[i];
                float b = v2[i];

                dot += a * b;
                norm1 += a * a;
                norm2 += b * b;
            }

            return dot / (float)(Math.Sqrt(norm1) * Math.Sqrt(norm2));
        }
        #endregion

    }
}
