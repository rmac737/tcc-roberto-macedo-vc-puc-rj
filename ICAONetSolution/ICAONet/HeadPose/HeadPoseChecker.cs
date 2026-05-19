using ICAONet.Enums;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Threading.Tasks;

namespace ICAONet.HeadPose
{
    public class HeadPoseChecker
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;
        private Utils _utils;
        public HeadPoseChecker()
        {
            _utils = new Utils();
            var so = new SessionOptions();
            so.InterOpNumThreads = Math.Max(1, Environment.ProcessorCount - 2);
            so.IntraOpNumThreads = 1;
            so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            so.EnableCpuMemArena = true;
            _session = new InferenceSession(_utils.LoadResource("ICAONet.HeadPose.sixdrepnet.onnx"), so);
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
        }

        public async Task<(bool, string)> Predict(Mat bgrImage)
        {
            return await Task.Run(() =>
            {
                if (bgrImage.Empty())
                    throw new ArgumentException("Imagem vazia");

                // 1. Pré-processamento
                using var resized = bgrImage.Resize(new OpenCvSharp.Size(224, 224));
                using var rgb = resized.CvtColor(ColorConversionCodes.BGR2RGB);
                var chw = ToCHWAndNormalize(rgb); // float[3*224*224]

                // 2. Tensor de entrada [1,3,224,224]
                var tensor = new DenseTensor<float>(chw, new[] { 1, 3, 224, 224 });

                // 3. Executa a inferência
                using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
                var output = results.First(x => x.Name == _outputName).AsEnumerable<float>().ToArray();

                if (output.Length != 9)
                    throw new Exception($"Saída inesperada: {output.Length} valores (esperado 6).");

                // 4. Converte para matriz de rotação (6D -> 3x3)
                var R = Ortho6DToR(output[0], output[1], output[2], output[3], output[4], output[5]);

                // 5. Extrai ângulos de Euler (pitch=X, yaw=Y, roll=Z)
                var (pitch, yaw, roll) = EulerFromR(R);

                // Ajuste de sinais (convenção humana)
                pitch = -pitch; // cima +
                yaw = -yaw;   // direita +
                roll = -roll;   // horário +

                string outText = $"Pitch: {pitch}; Yaw: {yaw}; Roll: {roll}";

                if ((yaw >= Config.HEAD_POSE_MIN_YAW && yaw <= Config.HEAD_POSE_MAX_YAW) &&
                    (pitch >= Config.HEAD_POSE_MIN_PITCH && pitch <= Config.HEAD_POSE_MAX_PITCH) &&
                    (roll >= Config.HEAD_POSE_MIN_ROLL && roll <= Config.HEAD_POSE_MAX_ROLL))
                {
                    return (true, $"{EnumICAOResult.PASSED}. {outText}.");
                }
                else
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED}. {outText}.");
                }
            });
        }

        private float[] ToCHWAndNormalize(Mat rgb)
        {
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };
            var data = new float[3 * 224 * 224];

            var idx = 0;
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < 224; y++)
                {
                    for (int x = 0; x < 224; x++)
                    {
                        Vec3b pix = rgb.At<Vec3b>(y, x);
                        float v = (c == 0 ? pix.Item0 : (c == 1 ? pix.Item1 : pix.Item2)) / 255f;
                        data[idx++] = (v - mean[c]) / std[c];
                    }
                }
            }
            return data;
        }

        private double[,] Ortho6DToR(float a1, float a2, float a3, float b1, float b2, float b3)
        {
            // 6D -> 3x3 rotação (Zhou et al.)
            var a = new Vector3(a1, a2, a3);
            var b = new Vector3(b1, b2, b3);

            var r1 = Vector3.Normalize(a);
            var bProj = b - Vector3.Dot(r1, b) * r1;
            var r2 = Vector3.Normalize(bProj);
            var r3 = Vector3.Cross(r1, r2);

            return new double[,] {
            { r1.X, r2.X, r3.X },
            { r1.Y, r2.Y, r3.Y },
            { r1.Z, r2.Z, r3.Z }
        };
        }
        private (double Pitch, double Yaw, double Roll) EulerFromR(double[,] R)
        {
            double r00 = R[0, 0], r01 = R[0, 1], r02 = R[0, 2];
            double r10 = R[1, 0], r11 = R[1, 1], r12 = R[1, 2];
            double r20 = R[2, 0], r21 = R[2, 1], r22 = R[2, 2];

            double sy = Math.Sqrt(r00 * r00 + r10 * r10);
            double x, y, z;
            if (sy > 1e-6)
            {
                x = Math.Atan2(r21, r22);
                y = Math.Atan2(-r20, sy);
                z = Math.Atan2(r10, r00);
            }
            else
            {
                x = Math.Atan2(-r12, r11);
                y = Math.Atan2(-r20, sy);
                z = 0;
            }

            const double k = 180.0 / Math.PI;
            return (x * k, y * k, z * k);
        }
    }
}
