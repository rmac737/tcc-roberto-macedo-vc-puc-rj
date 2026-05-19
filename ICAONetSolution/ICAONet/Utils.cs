using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet
{
    public class Utils
    {
        public byte[] LoadResource(string resourceName)
        {
            var asm = Assembly.GetExecutingAssembly();
            using var s = asm.GetManifestResourceStream(resourceName);
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            return ms.ToArray();
        }
        public string SerializeFloatArrayToStringBase64(float[] array)
        {
            string base64 = string.Empty;
            try
            {
                byte[] byteArray = new byte[array.Length * sizeof(float)];
                Buffer.BlockCopy(array, 0, byteArray, 0, byteArray.Length);
                base64 = Convert.ToBase64String(byteArray);
            }
            catch (Exception)
            {
                throw;
            }
            return base64;
        }
        public float[] DeserializeStringBase64ToFloatArray(string base64)
        {
            float[] ret = Array.Empty<float>();
            try
            {
                byte[] decodedBytes = Convert.FromBase64String(base64);
                float[] deserialized = new float[decodedBytes.Length / sizeof(float)];
                Buffer.BlockCopy(decodedBytes, 0, deserialized, 0, decodedBytes.Length);
                ret = deserialized;
            }
            catch (Exception)
            {
                throw;
            }
            return ret;
        }
        public Point2f[] To2D(Point3f[] pts3d)
        {
            return pts3d.Select(p => new Point2f(p.X, p.Y)).ToArray();
        }
    }

}
