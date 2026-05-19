using ICAONet.Enums;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace ICAONet.GazeEstimation
{
    public sealed class EyeGazeEstimator : IDisposable
    {
        private readonly InferenceSession _session;
        private Utils _utils;
        private readonly string _inputName;
        private readonly string _outputName;
        private readonly int _inW, _inH, _inC;

        // Normalização padrão PyTorch (ImageNet)
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        // NOVO: modos de desenho
        public enum GazeDrawMode
        {
            FromVector,   // usa (gx, gy, gz)
            FromYawPitch  // usa (yaw, pitch)
        }

        public EyeGazeEstimator(SessionOptions? opts = null)
        {
            _utils = new Utils();
            var so = new SessionOptions();
            so.InterOpNumThreads = Math.Max(1, Environment.ProcessorCount - 2);
            so.IntraOpNumThreads = 1;
            so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            so.EnableCpuMemArena = true;
            _session = new InferenceSession(_utils.LoadResource("ICAONet.GazeEstimation.resnet34_gaze.onnx"), so);

            // Descobrir input/output automaticamente
            var input = _session.InputMetadata.First();
            _inputName = input.Key;
            var inMd = input.Value;

            var output = _session.OutputMetadata.First();
            _outputName = output.Key;

            // Esperado: N x C x H x W (NCHW). Pegar H/W/C (C pode ser dinâmico mas usualmente 3)
            // Shapes podem ter dimensões -1 (dinâmicas). Pegamos valores positivos se houver.
            var shape = inMd.Dimensions.ToArray();
            // Tente detectar indices de C,H,W assumindo NCHW
            _inC = GetDimSafe(shape, 1, fallback: 3);
            _inH = GetDimSafe(shape, 2, fallback: 224);
            _inW = GetDimSafe(shape, 3, fallback: 224);
        }

        private static int GetDimSafe(int[] dims, int idx, int fallback)
        {
            if (idx < 0 || idx >= dims.Length) return fallback;
            var v = dims[idx];
            return v > 0 ? v : fallback;
        }

        public enum GazeDirection
        {
            Unknown, Center,
            Left, Right, Up, Down,
            UpLeft, UpRight, DownLeft, DownRight
        }

        public GazeDirection ClassifyDirection(GazeResult g,
            float yawThrDeg = 10f, float pitchThrDeg = 8f, float minVector = 0.05f)
        {
            // Se o vetor estiver muito pequeno, evita ruído
            float mag = MathF.Sqrt(g.GazeX * g.GazeX + g.GazeY * g.GazeY);
            if (mag < minVector) return GazeDirection.Center;

            bool left = g.YawDeg < -yawThrDeg;
            bool right = g.YawDeg > yawThrDeg;
            bool up = g.PitchDeg > pitchThrDeg;
            bool down = g.PitchDeg < -pitchThrDeg;

            if (!left && !right && !up && !down) return GazeDirection.Center;
            if (up && left) return GazeDirection.UpLeft;
            if (up && right) return GazeDirection.UpRight;
            if (down && left) return GazeDirection.DownLeft;
            if (down && right) return GazeDirection.DownRight;
            if (left) return GazeDirection.Left;
            if (right) return GazeDirection.Right;
            if (up) return GazeDirection.Up;
            if (down) return GazeDirection.Down;
            return GazeDirection.Center;
        }

        // Ângulo da seta em coordenadas de imagem (0° = direita, 90° = cima)
        public float DirectionAngleDeg(GazeResult g)
        {
            float dx = g.GazeX;
            float dy = -g.GazeY; // imagem: Y cresce para baixo, por isso o sinal
            if (MathF.Abs(dx) < 1e-6f && MathF.Abs(dy) < 1e-6f) return 0f;
            return MathF.Atan2(dy, dx) * (180f / MathF.PI);
        }

        public async Task<(bool,string)> Predict(Mat bgrImage)
        {
            return await Task.Run(() =>
            {
                if (bgrImage.Empty())
                    throw new ArgumentException("Imagem vazia.");

                // Pré-processamento: BGR -> RGB, resize para (W,H), float32 [0,1], normalização, NCHW
                using var rgb = bgrImage.CvtColor(ColorConversionCodes.BGR2RGB);
                using var resized = rgb.Resize(new Size(_inW, _inH), 0, 0, InterpolationFlags.Area);

                var chw = ToCHWAndNormalize(resized);

                // Cria tensor [1,C,H,W]
                // chw é o float[] já preenchido (C,H,W) normalizado
                var inputTensor = new DenseTensor<float>(chw, new[] { 1, _inC, _inH, _inW });

                var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

                using var outputs = _session.Run(inputs);
                var ov = outputs.First(o => o.Name == _outputName).Value;

                // Adaptação conforme formato de saída
                // Caso 1: 2 floats (pitch, yaw), radianos
                // Caso 2: 3 floats (gx, gy, gz)
                // Caso 3: logits/classe -> fallback: média ponderada simples (ingênua)
                float pitchRad, yawRad;
                float gx, gy, gz;

                if (ov is DenseTensor<float> tf)
                {
                    var arr = tf.ToArray();
                    if (arr.Length == 2)
                    {
                        pitchRad = arr[0]; yawRad = arr[1];
                        (gx, gy, gz) = YawPitchToVector(yawRad, pitchRad);
                    }
                    else if (arr.Length == 3)
                    {
                        gx = arr[0]; gy = arr[1]; gz = arr[2];
                        (pitchRad, yawRad) = VectorToYawPitch(gx, gy, gz);
                    }
                    else
                    {
                        // Fallback: tenta interpretar como logits 2D (pitch,yaw) discretizados.
                        // Estratégia simples: assume [N] onde N>=180 e mapeia índice máximo para yaw,
                        // e o segundo maior para pitch. (É um chute razoável quando não temos spec.)
                        var (p, y) = HeuristicFromLogits(arr);
                        pitchRad = p; yawRad = y;
                        (gx, gy, gz) = YawPitchToVector(yawRad, pitchRad);
                    }
                }
                else
                {
                    throw new NotSupportedException($"Tipo de saída não suportado: {ov.GetType().Name}");
                }

                //return new GazeResult
                //{
                //    PitchRad = pitchRad,
                //    YawRad = yawRad,
                //    PitchDeg = RadToDeg(pitchRad),
                //    YawDeg = RadToDeg(yawRad),
                //    GazeX = gx,
                //    GazeY = gy,
                //    GazeZ = gz
                //};

                string outText = $"Pitch: {Math.Abs(RadToDeg(pitchRad))}. Yaw: {Math.Abs(RadToDeg(yawRad))}";
                if(Math.Abs(RadToDeg(pitchRad)) <= 10 &&
                   Math.Abs(RadToDeg(yawRad)) <= 15)
                {
                    return (true, $"{EnumICAOResult.PASSED}. {outText}");
                }
                else
                {
                    return (false, $"{EnumICAOResult.NOT_PASSED}. {outText}");
                }
            });
        }

        public void DrawGaze(Mat imageBgr, Point origin, GazeResult gaze, int length = 60, int thickness = 2)
        {
            // Vetor 3D projetado em 2D simples: usar (gx, gy) como deslocamento na imagem.
            // Convenção: x -> direita, y -> baixo (imagem). gy normalmente sobe com pitch positivo,
            // então invertendo o sinal visualmente faz sentido.
            var end = new Point(
                origin.X + (int)(gaze.GazeX * length),
                origin.Y - (int)(gaze.GazeY * length)
            );

            Cv2.ArrowedLine(imageBgr, origin, end, Scalar.Red, thickness, LineTypes.AntiAlias, 0, 0.25);
        }

        private static float[] ToCHWAndNormalize(Mat rgb)
        {
            int h = rgb.Rows, w = rgb.Cols;
            var data = new float[3 * h * w];

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    // Lê pixel (r,g,b) como bytes
                    var pixel = rgb.At<Vec3b>(y, x);

                    float r = pixel.Item0 / 255f;
                    float g = pixel.Item1 / 255f;
                    float b = pixel.Item2 / 255f;

                    r = (r - Mean[0]) / Std[0];
                    g = (g - Mean[1]) / Std[1];
                    b = (b - Mean[2]) / Std[2];

                    int idx = y * w + x;
                    data[0 * h * w + idx] = r; // canal R
                    data[1 * h * w + idx] = g; // canal G
                    data[2 * h * w + idx] = b; // canal B
                }
            }
            return data;
        }

        // Conversões entre (pitch,yaw) e vetor 3D
        // Convenção típica:
        //  - pitch: rotação em torno de eixo X (para cima/baixo)
        //  - yaw: rotação em torno de eixo Y (para esquerda/direita)
        private static (float gx, float gy, float gz) YawPitchToVector(float yaw, float pitch)
        {
            // x = cos(pitch)*sin(yaw)
            // y = sin(pitch)
            // z = cos(pitch)*cos(yaw)
            float cp = (float)Math.Cos(pitch);
            float sp = (float)Math.Sin(pitch);
            float cy = (float)Math.Cos(yaw);
            float sy = (float)Math.Sin(yaw);
            float x = cp * sy;
            float y = sp;
            float z = cp * cy;

            // Normaliza
            float norm = (float)Math.Sqrt(x * x + y * y + z * z);
            if (norm > 1e-6f) { x /= norm; y /= norm; z /= norm; }
            return (x, y, z);
        }

        private static (float pitch, float yaw) VectorToYawPitch(float gx, float gy, float gz)
        {
            // Inversa:
            // pitch = asin(y)
            // yaw   = atan2(x, z)
            float pitch = (float)Math.Asin(Clamp(gy, -1f, 1f));
            float yaw = (float)Math.Atan2(gx, gz);
            return (pitch, yaw);
        }

        private static (float pitch, float yaw) HeuristicFromLogits(float[] logits)
        {
            // Fallback extremamente simples:
            // - Pega o índice do máx -> mapeia yaw em [-pi, pi]
            // - Pega o índice do 2º máx -> mapeia pitch em [-pi/2, pi/2]
            // OBS: Só para “quebrar o galho” se o modelo for classificatório sem spec.
            var idxs = Enumerable.Range(0, logits.Length)
                                 .OrderByDescending(i => logits[i])
                                 .Take(2)
                                 .ToArray();

            int idxYaw = idxs[0];
            int idxPitch = idxs[1];

            float yaw = MapToRange(idxYaw, logits.Length, -MathF.PI, MathF.PI);
            float pitch = MapToRange(idxPitch, logits.Length, -MathF.PI / 2f, MathF.PI / 2f);
            return (pitch, yaw);
        }

        private static float MapToRange(int idx, int n, float a, float b)
        {
            if (n <= 1) return (a + b) * 0.5f;
            float t = idx / (float)(n - 1);
            return a + t * (b - a);
        }

        private static float RadToDeg(float r) => r * (180f / MathF.PI);
        private static float Clamp(float v, float lo, float hi) => MathF.Max(lo, MathF.Min(hi, v));

        public void Dispose() => _session.Dispose();

        public void DrawGaze(
    Mat imageBgr,
    Point origin,                 // pixel coords no FRAME onde vai desenhar
    GazeResult gaze,
    int baseLength = 60,
    int thickness = 2,
    GazeDrawMode mode = GazeDrawMode.FromVector,
    bool scaleByForward = true,
    bool invertY = true,
    int minLength = 20            // comprimento mínimo pra não sumir
)
        {
            if (imageBgr.Empty()) return;

            // 0) bounds da origem
            if (origin.X < 0 || origin.Y < 0 || origin.X >= imageBgr.Cols || origin.Y >= imageBgr.Rows)
                return; // origem fora da imagem

            // 1) comprimento
            float forward = scaleByForward ? Clamp((gaze.GazeZ + 1f) * 0.5f, 0f, 1f) : 1f;
            int length = Math.Max(minLength, (int)(baseLength * (0.6f + 0.4f * forward)));

            // 2) direção
            float dx, dy;
            if (mode == GazeDrawMode.FromYawPitch)
            {
                float cp = (float)Math.Cos(gaze.PitchRad);
                float sp = (float)Math.Sin(gaze.PitchRad);
                float sy = (float)Math.Sin(gaze.YawRad);
                dx = cp * sy;
                dy = -sp;                  // pitch positivo = sobe -> y negativo na imagem
                if (!invertY) dy = -dy;    // opcional
            }
            else
            {
                dx = gaze.GazeX;
                dy = invertY ? -gaze.GazeY : gaze.GazeY;
            }

            // 3) normaliza (se zero, define default)
            float n = (float)Math.Sqrt(dx * dx + dy * dy);
            if (n < 1e-6f)
            {
                dx = 0f; dy = -1f; // seta pra cima como fallback
                n = 1f;
            }
            dx /= n; dy /= n;

            // 4) ponto final
            var end = new Point(
                origin.X + (int)(dx * length),
                origin.Y + (int)(dy * length)
            );

            // 5) clipe leve (evita coordenadas muito fora)
            end.X = Math.Clamp(end.X, 0, imageBgr.Cols - 1);
            end.Y = Math.Clamp(end.Y, 0, imageBgr.Rows - 1);

            // 6) desenha um ponto na origem pra você VER se está correto
            Cv2.Circle(imageBgr, origin, 3, Scalar.Red, -1, LineTypes.AntiAlias);

            // 7) seta
            Cv2.ArrowedLine(imageBgr, origin, end, Scalar.Red, thickness, LineTypes.AntiAlias, 0, 0.25);

            // 8) debug overlay (ângulos)
            string txt = $"yaw={gaze.YawDeg:F1}°, pitch={gaze.PitchDeg:F1}°";
            var orgTxt = new Point(Math.Min(origin.X + 6, imageBgr.Cols - 100), Math.Max(origin.Y - 8, 12));
            Cv2.PutText(imageBgr, txt, orgTxt, HersheyFonts.HersheySimplex, 0.4, Scalar.Lime, 1, LineTypes.AntiAlias);
        }

        // NOVO: desenhar a partir de yaw/pitch explicitamente (atalho)
        public void DrawGazeFromYawPitch(
            Mat imageBgr,
            Point origin,
            float yawRad,
            float pitchRad,
            int length = 60,
            int thickness = 2
        )
        {
            // dX = cos(pitch) * sin(yaw)
            // dY = -sin(pitch)
            float cp = (float)Math.Cos(pitchRad);
            float sp = (float)Math.Sin(pitchRad);
            float sy = (float)Math.Sin(yawRad);

            float dx = cp * sy;
            float dy = -sp;

            float n = (float)Math.Sqrt(dx * dx + dy * dy);
            if (n > 1e-6f) { dx /= n; dy /= n; }

            var end = new Point(
                origin.X + (int)(dx * length),
                origin.Y + (int)(dy * length)
            );

            Cv2.ArrowedLine(imageBgr, origin, end, Scalar.Lime, thickness, LineTypes.AntiAlias, 0, 0.25);
        }

        // OPCIONAL: "projeção pinhole" simples com focal length em px
        // Útil se você quiser aproximar uma perspectiva mais consistente com FOV da câmera.
        // fpx sugerido: ~ imagem.Width (ou ~0.8 * diagonal px)
        public void DrawGazePinhole(
            Mat imageBgr,
            Point origin,
            GazeResult gaze,
            float fpx,                // focal length aproximado em pixels
            int lengthPx = 120,       // comprimento virtual em 3D projetado em 2D
            int thickness = 2
        )
        {
            // Vetor direção 3D (normalizado)
            float x = gaze.GazeX, y = gaze.GazeY, z = gaze.GazeZ;
            // definimos um “ponto 3D” a frente do olho: P = origin3D + L * dir
            // como não temos depth real, fazemos uma projeção diferencial:
            // deslocamento 2D ≈ f * (x/z, y/z) * k, com proteção para z pequeno
            float zSafe = MathF.Max(0.15f, MathF.Abs(z)); // evita blow-up
            float u = (fpx * x / zSafe);
            float v = (fpx * (true ? -y : y) / zSafe);

            // normaliza e escala para ter um controle de comprimento visual
            float wn = (float)Math.Sqrt(u * u + v * v);
            if (wn > 1e-6f) { u = u / wn * lengthPx; v = v / wn * lengthPx; }

            var end = new Point(
                origin.X + (int)u,
                origin.Y + (int)v
            );

            Cv2.ArrowedLine(imageBgr, origin, end, Scalar.Cyan, thickness, LineTypes.AntiAlias, 0, 0.25);
        }
    }

    public sealed class GazeResult
    {
        public float PitchRad { get; set; }
        public float YawRad { get; set; }
        public float PitchDeg { get; set; }
        public float YawDeg { get; set; }
        public float GazeX { get; set; }
        public float GazeY { get; set; }
        public float GazeZ { get; set; }
    }
}
