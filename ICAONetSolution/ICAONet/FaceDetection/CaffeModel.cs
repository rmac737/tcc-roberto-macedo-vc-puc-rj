using Microsoft.Extensions.ObjectPool;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.FaceDetection
{
    public class CaffeModel
    {
        private ObjectPool<Net> _netPool;
        private Utils _utils;
        public CaffeModel()
        {
            _utils = new Utils();
            var provider = new DefaultObjectPoolProvider { MaximumRetained = Environment.ProcessorCount * 2 };
            _netPool = provider.Create(new OpenCvNetPooledFaceDetectionPolicy(_utils.LoadResource("ICAONet.FaceDetection.deploy.prototxt"), _utils.LoadResource("ICAONet.FaceDetection.res10_300x300_ssd_iter_140000.caffemodel")));
        }

        public (byte[], Rect rect) GetFaceCropped(byte[] imagem)
        {
            List<(byte[], Rect rect)> ret = new List<(byte[], Rect)>();
            var net = _netPool.Get();

            try
            {
                using Mat image = Cv2.ImDecode(imagem, ImreadModes.AnyColor);

                Mat blob = CvDnn.BlobFromImage(image, 1.0, new OpenCvSharp.Size(300, 300), new Scalar(104, 177, 123), false, false);

                net!.SetInput(blob);
                Mat detection = net.Forward();

                int detections = detection.Size(2);

                for (int i = 0; i < detections; i++)
                {
                    float confidence = detection.At<float>(0, 0, i, 2);
                    if (confidence > 0.5f)
                    {
                        int x1 = (int)(detection.At<float>(0, 0, i, 3) * image.Cols);
                        int y1 = (int)(detection.At<float>(0, 0, i, 4) * image.Rows);
                        int x2 = (int)(detection.At<float>(0, 0, i, 5) * image.Cols);
                        int y2 = (int)(detection.At<float>(0, 0, i, 6) * image.Rows);

                        // Medidas originais
                        int w = x2 - x1;
                        int h = y2 - y1;

                        // Centro do retângulo
                        double cx = x1 + w / 2.0;
                        double cy = y1 + h / 2.0;

                        // ==============================
                        // (A) Retângulo expandido 1.2x mantendo a razão
                        // ==============================
                        double scale = 1.2;
                        int newW = (int)Math.Round(w * scale);
                        int newH = (int)Math.Round(h * scale);

                        // // ==============================
                        // // (B) Alternativa: corte QUADRADO 1.2x (descomente essas 3 linhas e comente as 2 acima)
                        // int side = (int)Math.Round(Math.Max(w, h) * 1.2);
                        // int newW = side;
                        // int newH = side;
                        // // ==============================

                        int newX = (int)Math.Round(cx - newW / 2.0);
                        int newY = (int)Math.Round(cy - newH / 2.0);

                        // Clampeia aos limites da imagem
                        newX = Math.Max(0, Math.Min(newX, image.Cols - 1));
                        newY = Math.Max(0, Math.Min(newY, image.Rows - 1));

                        // Ajusta largura/altura para não ultrapassar a borda
                        if (newX + newW > image.Cols) newW = image.Cols - newX;
                        if (newY + newH > image.Rows) newH = image.Rows - newY;

                        // Evita dimensões inválidas
                        newW = Math.Max(1, newW);
                        newH = Math.Max(1, newH);

                        // Cria o retângulo final
                        Rect rect = new Rect(newX, newY, newW, newH);

                        Cv2.Rectangle(image, rect, Scalar.Black, 1);

                        Mat faceCrop = new Mat(image, rect);
                        ret.Add((faceCrop.ToBytes(),rect));
                    }
                }

                if (ret.Count == 0)
                {
                    throw new Exception("Não foi encontrado uma face.");
                }
                else if (ret.Count > 1)
                {
                    throw new Exception("Mais de uma face encontrada.");
                }
                else
                {
                    return ret.FirstOrDefault();
                }
            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                _netPool.Return(net);
            }
        }

    }
}
