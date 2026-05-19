using ICAONet.Enums;
using ICAONet.FaceDetection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

namespace ICAONet.Landmarks
{
    public class FaceMesh468 : IDisposable
    {
        private readonly InferenceSession _session;
        private Utils _utils;
        private readonly string _imgInputName;
        private readonly string? _cropX1, _cropY1, _cropW, _cropH;
        private readonly string _lmkOutName;
        private readonly string? _scoreName;
        private readonly int _inW, _inH;
        
        public FaceMesh468()
        {
            _utils = new Utils();
            var so = new SessionOptions();
            so.InterOpNumThreads = Math.Max(1, Environment.ProcessorCount - 2);
            so.IntraOpNumThreads = 1;
            so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            so.EnableCpuMemArena = true;
            _session = new InferenceSession(_utils.LoadResource("ICAONet.Landmarks.face_mesh_Nx3x192x192_post.onnx"), so);

            var inputs = _session.InputMetadata.Keys.ToArray();
            var outputs = _session.OutputMetadata.Keys.ToArray();

            // Descobre input NCHW
            _imgInputName = inputs.First(k => _session.InputMetadata[k].Dimensions.Length == 4);
            var dims = _session.InputMetadata[_imgInputName].Dimensions; // [N,3,H,W] ou [-1,3,192,192]
            _inH = dims[2] > 0 ? dims[2] : 192;
            _inW = dims[3] > 0 ? dims[3] : 192;

            // Detecta se é variante "post" (tem tensores de crop)
            _cropX1 = inputs.FirstOrDefault(k => k.Contains("crop_x1", StringComparison.OrdinalIgnoreCase));
            _cropY1 = inputs.FirstOrDefault(k => k.Contains("crop_y1", StringComparison.OrdinalIgnoreCase));
            _cropW = inputs.FirstOrDefault(k => k.Contains("crop_width", StringComparison.OrdinalIgnoreCase));
            _cropH = inputs.FirstOrDefault(k => k.Contains("crop_height", StringComparison.OrdinalIgnoreCase));

            // Escolhe um output plausível de landmarks
            _lmkOutName = outputs.FirstOrDefault(k => k.Contains("landmark", StringComparison.OrdinalIgnoreCase))
                          ?? outputs.FirstOrDefault(k => k.Contains("final", StringComparison.OrdinalIgnoreCase))
                          ?? outputs.First();

            _scoreName = outputs.FirstOrDefault(k => k.Contains("score", StringComparison.OrdinalIgnoreCase));
        }
        public async Task<(Point3f[] Keypoints, float Score, OutputMode Mode)> Run(Mat bgrImage, float marginRatio = 0.25f)
        {
            return await Task.Run(() =>
            {
                var (image, bbox) = new CaffeModel().GetFaceCropped(bgrImage.ToBytes());
                if (bbox.Width == 0) throw new Exception("Nenhuma face detectada.");

                var (cropRect, cx1, cy1, cw, ch) = ExpandWithMargin(bbox, bgrImage.Width, bgrImage.Height, marginRatio);

                using var faceRoi = new Mat(bgrImage, cropRect);
                using var resized = faceRoi.Resize(new Size(_inW, _inH));
                Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

                var chw = new DenseTensor<float>(new[] { 1, 3, _inH, _inW });
                var idxr = resized.GetGenericIndexer<Vec3b>();

                for (int y = 0; y < _inH; y++)
                {
                    for (int x = 0; x < _inW; x++)
                    {
                        var px = idxr[y, x]; // já está em RGB (você converteu acima)
                        chw[0, 0, y, x] = px.Item0 / 255f; // R
                        chw[0, 1, y, x] = px.Item1 / 255f; // G
                        chw[0, 2, y, x] = px.Item2 / 255f; // B
                    }
                }

                bool hasPostCrops = _cropX1 != null && _cropY1 != null && _cropW != null && _cropH != null;
                var io = new System.Collections.Generic.List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor(_imgInputName, chw)
            };

                if (hasPostCrops)
                {
                    var tX1 = new DenseTensor<int>(new[] { 1, 1 }); tX1[0, 0] = cx1;
                    var tY1 = new DenseTensor<int>(new[] { 1, 1 }); tY1[0, 0] = cy1;
                    var tW = new DenseTensor<int>(new[] { 1, 1 }); tW[0, 0] = cw;
                    var tH = new DenseTensor<int>(new[] { 1, 1 }); tH[0, 0] = ch;

                    io.Add(NamedOnnxValue.CreateFromTensor(_cropX1!, tX1));
                    io.Add(NamedOnnxValue.CreateFromTensor(_cropY1!, tY1));
                    io.Add(NamedOnnxValue.CreateFromTensor(_cropW!, tW));
                    io.Add(NamedOnnxValue.CreateFromTensor(_cropH!, tH));
                }

                using var results = _session.Run(io);

                // ---- SCORE ----
                float score = 1f;
                if (_scoreName != null && results.Any(r => r.Name == _scoreName))
                {
                    var sv = results.First(r => r.Name == _scoreName).Value;

                    if (sv is Tensor<float> tScoreF)
                        score = tScoreF.ToArray().FirstOrDefault();
                    else if (sv is Tensor<int> tScoreI)
                        score = tScoreI.ToArray().FirstOrDefault();
                }

                // ---- LANDMARKS ----
                var ov = results.First(r => r.Name == _lmkOutName).Value;

                // 1) Se vier INT: assumimos pixels absolutos
                if (ov is Tensor<int> tLmkI)
                {
                    var pts = TensorToPointsInt(tLmkI);
                    return (pts, score, OutputMode.PixelsAbsolutos);
                }

                // 2) Se vier FLOAT: auto-detecta o range
                if (ov is Tensor<float> tLmkF)
                {
                    var raw = TensorToRaw(tLmkF, out int ptsCount, out int coordDim);
                    var (mode, pts) = InterpretFloats(raw, ptsCount, coordDim, cropRect);
                    return (pts, score, mode);
                }

                throw new InvalidOperationException("Saída de landmarks não é tensor float/int.");
            });
        }
        private (OutputMode mode, Point3f[] pts) InterpretFloats(float[] flat, int ptsCount, int coordDim, Rect crop)
        {
            // Examina faixa dos X/Y
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;

            for (int i = 0; i < ptsCount; i++)
            {
                float x = flat[i * coordDim + 0];
                float y = flat[i * coordDim + 1];
                minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
            }

            // Pixels prováveis (valores bem maiores que 1)
            bool looksPixels = (maxX > 1.5f || maxY > 1.5f);
            if (looksPixels)
            {
                var ptsPx = new Point3f[ptsCount];
                for (int i = 0; i < ptsCount; i++)
                {
                    float x = flat[i * coordDim + 0];
                    float y = flat[i * coordDim + 1];
                    float z = coordDim >= 3 ? flat[i * coordDim + 2] : 0f; // pode já estar em px ou relativo
                    ptsPx[i] = new Point3f(x, y, z);
                }
                return (OutputMode.PixelsAbsolutos, ptsPx);
            }

            // [-1,1] ou [0,1]?
            bool inside01 = (minX >= -0.05f && maxX <= 1.05f && minY >= -0.05f && maxY <= 1.05f);
            bool aroundMinus11 = (minX >= -1.2f && maxX <= 1.2f && minY >= -1.2f && maxY <= 1.2f);
            OutputMode mode = inside01 ? OutputMode.NormalizadoZeroUm : OutputMode.NormalizadoMenosUmUm;

            var pts = new Point3f[ptsCount];
            for (int i = 0; i < ptsCount; i++)
            {
                float xN = flat[i * coordDim + 0];
                float yN = flat[i * coordDim + 1];
                float zN = coordDim >= 3 ? flat[i * coordDim + 2] : 0f;

                float xImg, yImg;
                if (mode == OutputMode.NormalizadoZeroUm)
                {
                    xImg = crop.X + xN * crop.Width;
                    yImg = crop.Y + yN * crop.Height;
                }
                else
                {
                    xImg = crop.X + ((xN + 1f) * 0.5f) * crop.Width;
                    yImg = crop.Y + ((yN + 1f) * 0.5f) * crop.Height;
                }
                float zImg = zN * crop.Width;
                pts[i] = new Point3f(xImg, yImg, zImg);
            }
            return (mode, pts);
        }
        private float[] TensorToRaw(Tensor<float> t, out int ptsCount, out int coordDim)
        {
            var dims = t.Dimensions.ToArray(); // ex: [1,468,3] ou [468,3]
            coordDim = dims[^1];
            ptsCount = dims[^2];

            var flat = t.ToArray(); // row-major
                                    // Se vier com dimensão extra N=1, já está ok: layout [pts,coord] no final
            return flat;
        }
        private Point3f[] TensorToPointsInt(Tensor<int> t)
        {
            var dims = t.Dimensions.ToArray(); // [1,468,3] ou [468,3]
            int coordDim = dims[^1];
            int ptsCount = dims[^2];
            var flat = t.ToArray();

            var pts = new Point3f[ptsCount];
            for (int i = 0; i < ptsCount; i++)
            {
                int x = flat[i * coordDim + 0];
                int y = flat[i * coordDim + 1];
                int z = coordDim >= 3 ? flat[i * coordDim + 2] : 0;
                pts[i] = new Point3f(x, y, z);
            }
            return pts;
        }
        private (Rect rect, int x1, int y1, int w, int h) ExpandWithMargin(Rect box, int imgW, int imgH, float m)
        {
            int cx1 = Math.Max(0, (int)Math.Floor(box.X - box.Width * m));
            int cy1 = Math.Max(0, (int)Math.Floor(box.Y - box.Height * m));
            int cx2 = Math.Min(imgW - 1, (int)Math.Ceiling(box.Right + box.Width * m));
            int cy2 = Math.Min(imgH - 1, (int)Math.Ceiling(box.Bottom + box.Height * m));
            int cw = Math.Max(1, cx2 - cx1);
            int ch = Math.Max(1, cy2 - cy1);
            return (new Rect(cx1, cy1, cw, ch), cx1, cy1, cw, ch);
        }
        public void Dispose() => _session?.Dispose();
    }
}
