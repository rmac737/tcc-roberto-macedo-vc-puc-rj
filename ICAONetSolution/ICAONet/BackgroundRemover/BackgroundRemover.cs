using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;

namespace ICAONet.Background
{
    public class BackgroundRemover
    {
        private readonly InferenceSession _session;
        private Utils _utils;

        public BackgroundRemover()
        {
            _utils = new Utils();
            var so = new SessionOptions();
            so.InterOpNumThreads = Math.Max(1, Environment.ProcessorCount - 2);
            so.IntraOpNumThreads = 1;
            so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            so.EnableCpuMemArena = true;
            _session = new InferenceSession(_utils.LoadResource("ICAONet.BackgroundRemover.rvm_resnet50.onnx"),so);
        }
        public async Task<byte[]> RemoverFundo(Mat img)
        {
            return await Task.Run(() =>
            {
                // Carrega imagem
                int h = img.Rows - (img.Rows % 32);
                int w = img.Cols - (img.Cols % 32);
                Cv2.Resize(img, img, new OpenCvSharp.Size(w, h));

                var imgData = MatToTensor(img); // você precisa escrever esse método (ver abaixo)

                // 📦 Inicializar estados dinamicamente
                List<NamedOnnxValue> CreateInitialStates(int h, int w, float ratio = 0.25f)
                {
                    int h_ds = (int)Math.Ceiling(h * ratio);
                    int w_ds = (int)Math.Ceiling(w * ratio);

                    int r1h = (int)Math.Ceiling(h_ds / 2f);
                    int r1w = (int)Math.Ceiling(w_ds / 2f);
                    int r2h = (int)Math.Ceiling(h_ds / 4f);
                    int r2w = (int)Math.Ceiling(w_ds / 4f);
                    int r3h = (int)Math.Ceiling(h_ds / 8f);
                    int r3w = (int)Math.Ceiling(w_ds / 8f);
                    int r4h = (int)Math.Ceiling(h_ds / 16f);
                    int r4w = (int)Math.Ceiling(w_ds / 16f);

                    return new()
                {
                    NamedOnnxValue.CreateFromTensor("r1", new DenseTensor<float>(new[] {1, 16, r1h, r1w})),
                    NamedOnnxValue.CreateFromTensor("r2", new DenseTensor<float>(new[] {1, 32, r2h, r2w})),
                    NamedOnnxValue.CreateFromTensor("r3", new DenseTensor<float>(new[] {1, 64, r3h, r3w})),
                    NamedOnnxValue.CreateFromTensor("r4", new DenseTensor<float>(new[] {1,128, r4h, r4w}))
                };
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("src", imgData)
                };
                inputs.AddRange(CreateInitialStates(h, w));   // ratio já default 0.25

                using var results = _session.Run(inputs);

                var pha = results.First(x => x.Name == "pha").AsEnumerable<float>().ToArray();
                // fgr, r1_out... também disponíveis

                Console.WriteLine("Inferência concluída");

                // Supondo que você já tem `fgr` e `pha` como float[] de shape [1, 3, H, W] e [1, 1, H, W]
                int height = img.Rows;
                int width = img.Cols;

                // Extrair `fgr` e `pha` dos resultados
                var fgrTensor = results.First(x => x.Name == "fgr").AsEnumerable<float>().ToArray();
                var phaTensor = results.First(x => x.Name == "pha").AsEnumerable<float>().ToArray();

                Mat output = new Mat(height, width, MatType.CV_8UC3);

                unsafe
                {
                    byte* ptr = (byte*)output.DataPointer;

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int index = y * width + x;

                            float alpha = phaTensor[index]; // [1,1,H,W]
                            byte white = 255;

                            // fgr [1,3,H,W]
                            float r = fgrTensor[0 * height * width + index];
                            float g = fgrTensor[1 * height * width + index];
                            float b = fgrTensor[2 * height * width + index];

                            byte rb = (byte)(r * alpha * 255 + (1 - alpha) * white);
                            byte gb = (byte)(g * alpha * 255 + (1 - alpha) * white);
                            byte bb = (byte)(b * alpha * 255 + (1 - alpha) * white);

                            int pixelIndex = (int)(y * output.Step() + x * 3);
                            ptr[pixelIndex + 0] = bb;
                            ptr[pixelIndex + 1] = gb;
                            ptr[pixelIndex + 2] = rb;
                        }
                    }
                }

                // Salvar a imagem final
                using var final = new Mat();
                Cv2.Resize(output, final, new OpenCvSharp.Size(img.Width, img.Height));
                return final.ToBytes();
            });   
        }
        private DenseTensor<float> MatToTensor(Mat img)
        {
            int height = img.Rows;
            int width = img.Cols;

            // Converter BGR para RGB (opcional, depende do modelo)
            Mat rgb = new Mat();
            Cv2.CvtColor(img, rgb, ColorConversionCodes.BGR2RGB);

            // Criar tensor 4D [1,3,H,W]
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            unsafe
            {
                byte* ptr = (byte*)rgb.DataPointer;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = (int)(y * rgb.Step() + x * rgb.ElemSize());

                        // rgb channels
                        byte r = ptr[pixelIndex + 0];
                        byte g = ptr[pixelIndex + 1];
                        byte b = ptr[pixelIndex + 2];

                        // Normalize para [0,1] e preencha no tensor
                        tensor[0, 0, y, x] = r / 255f;
                        tensor[0, 1, y, x] = g / 255f;
                        tensor[0, 2, y, x] = b / 255f;
                    }
                }
            }

            return tensor;
        }
    }
}
