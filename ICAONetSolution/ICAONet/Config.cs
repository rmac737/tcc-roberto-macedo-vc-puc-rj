using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace ICAONet
{
    public static class Config
    {
        public static int MAX_FACES { get; private set; } = 1;

        //HEADPOSE THRESHOLDS
        public static int HEAD_POSE_MIN_YAW { get; private set; } = -20;
        public static int HEAD_POSE_MAX_YAW { get; private set; } = 20;

        public static int HEAD_POSE_MIN_PITCH { get; private set; } = -10;
        public static int HEAD_POSE_MAX_PITCH { get; private set; } = 10;

        public static int HEAD_POSE_MIN_ROLL { get; private set; } = -5;
        public static int HEAD_POSE_MAX_ROLL { get; private set; } = 5;

        
        //EYES OPEN, MOUTH OPEN THREHSOLDS

        public static double MOUTH_THRESHOLD { get; private set; } = 0.5;
        public static double EYES_THRESHOLD { get; private set; }=  0.09;

        //IMAGE AND FACE DIMENSION THRESHOLDS

        public static double MIN_WIDTH_HEIGHT_RATIO { get; private set; } = 0.74;
        public static double MAX_WIDTH_HEIGHT_RATIO { get; private set; } = 0.80;

        public static double MIN_LEFT_DISTANCE { get; private set; } = 0.45;
        public static double MAX_LEFT_DISTANCE { get; private set; } = 0.55;

        public static double MIN_TOP_DISTANCE { get; private set; } = 0.30;
        public static double MAX_TOP_DISTANCE { get; private set; } = 0.50;

        public static double MIN_WIDTH_HEAD { get; private set; } = 0.50;
        public static double MAX_WIDTH_HEAD { get; private set; } = 0.75;

        public static double MIN_HEIGHT_HEAD { get; private set; } = 0.60;
        public static double MAX_HEIGHT_HEAD { get; private set; } = 0.90;

        public static double MINIMUM_IED { get; private set; } = 0.50;

        //EXPOSURE CONSTANTS

        public static double AVG_DARK_THRESHOLD { get; private set; } = 0.0055;
        public static double MAX_DARK_THRESHOLD { get; private set; } = 0.015;

        public static double AVG_LIGHT_THRESHOLD { get; private set; } = 0.0035;
        public static double MAX_LIGHT_THRESHOLD { get; private set; } = 0.015;

        //PIXELATION THRESHOLDS

        public static int PIXELATED_MIN_THRESHOLD { get; private set; } = 19;
        public static int PIXELATED_MAX_THRESHOLD { get; private set; } = 106;
        public static double PIXELATED_THRESHOLD { get; private set; } = 0.6;

        //EMOTION DETECTION THRESHOLD
        public static double MAX_RATIO_EMOTION { get; private set; } = 0.35;

        //PARSER THRESHOLDS

        //SHOULDER CHECK

        public static double MAX_SHOULDER_PIXEL_RATIO { get; private set; } = 0.55;
        public static int MAX_SHOULDER_Y_DISTANCE { get; private set; } = 5;

        //SATURATION CHECK

        public static double OVERSATURATION_THRESHOLD { get; private set; } = 0.2;
        public static int UNDERSATURATION_THRESHOLD { get; private set; } = 6;

        //FACE ILLUMINATION

        public static int MAX_BRIGHT_LIGHT { get; private set; } = 5;
        public static int MAX_DARK_LIGHT { get; private set; } = 70;

        //BACKGROUND CHECK

        public static int MAX_EDGES_THRESHOLD { get; private set; } = 900000000;
        public static int AVG_VARIANCE_THRESHOLD { get; private set; } = 500;
        public static double HOMOGENEOUS_PROPORTION_THRESHOLD { get; private set; } = 0.90;
        public static int SUPERPIXEL_VARIANCE_THRESHOLD { get; private set; } = 45;
        public static double MAX_LIGHT_DARK_SUN { get; private set; } = 0.004;

        //COMPUTER VISION THRESHOLDS

        public static int MINIMUM_FOCUS_THRESHOLD { get; private set; } = 300;

        //POSTERIZATION

        public static int MAX_GAPS_THRESHOLD { get; private set; } = 210;
        public static double GAP_HISTOGRAM_THRESHOLD { get; private set; } = 0.001;


        //PIXELIZATION

        public static double PIXEL_SCORE_THRESHOLD { get; private set; } = 0.5;
        public static int MIN_PIXEL_SCORE { get; private set; } = 10;
        public static int MAX_PIXEL_SCORE { get; private set; } = 110;

        //LANDMARK THRESHOLDS

        //FACE ILLUMINATION

        public static double MIN_COLOR_RATIO_THRESOLD { get; private set; } = 0.5;

        //MAKE UP

        public static double MAKEUP_HIGH_HUE_THRESHOLD { get; private set; } = 0.15;

        public static int MAKEUP_DISTANCE_THRESHOLD { get; private set; } = 14;

        //GAZE ESTIMATION THRESHOLDS

        public static double MAXIMUM_RIGHT_THRESHOLD { get; private set; } = -0.15;
        public static double MAXIMUM_LEFT_THRESHOLD { get; private set; } = 0.08;
    }
}
