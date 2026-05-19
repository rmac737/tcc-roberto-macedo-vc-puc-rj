using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet.Enums
{
    public enum EnumItemsVerificados
    {
        HEAD_WITHOUT_COVERING = 0,
        EYES_OPEN = 1,
        NO_SUNGLASSES = 2,
        NO_POSTERIZATION = 3,
        GAZE_IN_CAMERA = 4,
        NEUTRAL_EXPRESSION = 5,
        IN_FOCUS_PHOTO = 6,
        CORRECT_EXPOSURE = 7,
        NO_LIGHT_MAKEUP = 8,
        NO_PIXELATION = 9,
        FRONTAL_POSE = 10,
        CORRECT_SATURATION = 11,
        UNIFORM_BACKGROUND = 12,
        UNIFORM_FACE_LIGHTING = 13
    }

    public enum LandmarkChecksMap
    {
        EYES_OPEN,
        NEUTRAL_EXPRESSION,
        NO_LIGHT_MAKEUP,
        UNIFORM_FACE_LIGHTING
    }

    public enum ParserChecksMap
    {
        HEAD_WITHOUT_COVERING,
        NO_SUNGLASSES,
        FRONTAL_POSE,
        CORRECT_SATURATION,
        UNIFORM_BACKGROUND,
        UNIFORM_FACE_LIGHTING
    }
}
