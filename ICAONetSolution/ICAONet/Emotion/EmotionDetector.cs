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

namespace ICAONet.Emotion
{
    public class EmotionDetector
    {
        private readonly InferenceSession _session;
        private Utils _utils;
        private readonly string _inputName;
        private readonly string _outputName;
        private readonly int _inH;
        private readonly int _inW;

        // labels FER+
        private static readonly string[] LABELS_8 = new[]
        {
            "anger","contempt","disgust","fear","happiness","neutral","sadness","surprise"
        };

        // Normalização padrão ImageNet
        private static readonly float[] MEAN = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] STD = { 0.229f, 0.224f, 0.225f };

        public EmotionDetector() 
        {   
            _utils = new Utils();
            var so = new SessionOptions();
            so.InterOpNumThreads = Math.Max(1, Environment.ProcessorCount - 2);
            so.IntraOpNumThreads = 1;
            so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            so.EnableCpuMemArena = true;
            _session = new InferenceSession(_utils.LoadResource("ICAONet.Emotion.enet_b2_8.onnx"), so);
            // Descobre I/O
            var input = _session.InputMetadata.First();
            _inputName = input.Key;
            var dims = input.Value.Dimensions; // [N,C,H,W] (H/W possivelmente -1)
            _inH = dims.Length >= 4 && dims[2] > 0 ? dims[2] : 260;
            _inW = dims.Length >= 4 && dims[3] > 0 ? dims[3] : 260;

            var output = _session.OutputMetadata.First();
            _outputName = output.Key;
        }
        public async Task<(bool,string)> Predict(Mat bgr)
        {
            return await Task.Run(() =>
            {
                // Pré-processa (BGR->RGB, resize 260x260, [0..1], normalize, CHW)
                var tensor = PreprocessToTensor(bgr, _inW, _inH);

                // Inferência
                var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };
                using var results = _session.Run(inputs);
                var logits = results.First(v => v.Name == _outputName).AsEnumerable<float>().ToArray();

                // Softmax
                var probs = Softmax(logits);

                // Labels (espera 8)
                var labels = LABELS_8;
                if (probs.Length != labels.Length)
                {
                    // fallback (p. ex., se você carregar por engano um enet_b2_7.onnx)
                    labels = labels.Take(probs.Length).ToArray();
                }

                int top = Array.IndexOf(probs, probs.Max());

                string topLabelTemp = labels[top];
                double probsTemp = probs[top];

                if((topLabelTemp.Equals("happiness") || topLabelTemp.Equals("surprise")) && (probsTemp > 0.30))
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED}. Expressão não neutra. Detectado: {topLabelTemp}");
                }
                else
                {
                    return (true, $"{EnumICAOResult.PASSED}. Expressão neutra OK.");
                }
            });
        }
        private DenseTensor<float> PreprocessToTensor(Mat bgr, int width, int height)
        {
            // 1) BGR -> RGB
            using var rgb = new Mat();
            Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);

            // 2) Resize
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(width, height), 0, 0, InterpolationFlags.Area);

            // 3) Copia para array CHW float32 normalizado (evita fazer "Mat - float")
            int C = 3, H = height, W = width;
            var data = new float[1 * C * H * W];

            // Acesso rápido linha a linha
            for (int y = 0; y < H; y++)
            {
                for (int x = 0; x < W; x++)
                {
                    var rgbPix = resized.At<Vec3b>(y, x); // R,G,B em ordem (já convertida)
                    float r = rgbPix.Item0 / 255f;
                    float g = rgbPix.Item1 / 255f;
                    float b = rgbPix.Item2 / 255f;

                    // Normalização por canal
                    r = (r - MEAN[0]) / STD[0];
                    g = (g - MEAN[1]) / STD[1];
                    b = (b - MEAN[2]) / STD[2];

                    int baseIdx = y * W + x;
                    data[0 * H * W + baseIdx] = r;            // C=0 (R)
                    data[1 * H * W + baseIdx] = g;            // C=1 (G)
                    data[2 * H * W + baseIdx] = b;            // C=2 (B)
                }
            }

            var tensor = new DenseTensor<float>(data, new[] { 1, C, H, W });
            return tensor;
        }
        private float[] Softmax(float[] logits)
        {
            // estabilidade numérica
            float max = logits.Max();
            var exps = logits.Select(v => MathF.Exp(v - max)).ToArray();
            float sum = exps.Sum();
            if (sum <= 0) sum = 1f;
            return exps.Select(v => v / sum).ToArray();
        }
        public void Dispose() => _session.Dispose();
    }

    public sealed record EmotionResult(
        string[] Labels,
        float[] Probs,
        string TopLabel,
        float TopProb
    );
}
