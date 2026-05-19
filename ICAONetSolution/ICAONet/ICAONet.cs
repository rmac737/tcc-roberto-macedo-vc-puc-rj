using Clipper2Lib;
using ICAONet.Background;
using ICAONet.ClosedMouth;
using ICAONet.DTO;
using ICAONet.Emotion;
using ICAONet.Enums;
using ICAONet.EyesOpen;
using ICAONet.FaceCrop;
using ICAONet.FaceDetection;
using ICAONet.GazeEstimation;
using ICAONet.GlassDetection;
using ICAONet.HeadCoverDetection;
using ICAONet.HeadPose;
using ICAONet.ImageQuality;
using ICAONet.Landmarks;
using ICAONet.MakeUpDetection;
using ICAONet.RedEyes;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ICAONet
{
    public class ICAONet
    {
        private Utils _utils;
        private QualityChecker _qualityChecker;
        public ICAONet()
        {
            _utils = new Utils();
            _qualityChecker = new QualityChecker();
        }

        public async Task<EvaluationICAOResultDTO> Evaluate(ICAONetParams iCAONetParams)
        {
            EvaluationICAOResultDTO evaluationResult = new EvaluationICAOResultDTO();
            evaluationResult.OriginalImage = iCAONetParams.Image;

            #region "Passo 1 - Validação da imagem de entrada"
            evaluationResult.IsImageOkToBeProcessed = await _qualityChecker.IsImageOkToBeProcessed(iCAONetParams.Image);
            #endregion

            #region "Passo 2 - Criando uma instância de MAT da imagem original"
            using Mat originalImage = Cv2.ImDecode(iCAONetParams.Image, ImreadModes.AnyColor);
            #endregion

            #region "Passo 3 - Crop a face"
            CaffeModel caffeModel = new CaffeModel();
            var (faceCropped, rectangle) = caffeModel.GetFaceCropped(iCAONetParams.Image);
            Mat matFaceCroppedForEvaluation = Cv2.ImDecode(faceCropped, ImreadModes.AnyColor);
            #endregion

            #region "Passo 4 - Extrair os pontos da face a partir da face recortada"
            FaceMesh468 faceMesh468 = new FaceMesh468();
            var (keypoints, score, mode) = await faceMesh468.Run(originalImage);
            var (keypointsForEvaluation, scoreForEvaluation, modeForEvaluation) = await faceMesh468.Run(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 5 - Remoção do fundo 5.1"
            byte[] originalImageWithoutBackground = Array.Empty<byte>();
            Mat croppedICAOFormat = new Mat();

            BackgroundRemover backgroundRemover = new BackgroundRemover();
            FaceCropper faceCropper = new FaceCropper();

            //Imagem original sem o fundo
            byte[] tempImageWithoubackgroundIcao = await backgroundRemover.RemoverFundo(originalImage);
            Mat imagemMatWithoutBackground = Cv2.ImDecode(tempImageWithoubackgroundIcao, ImreadModes.AnyColor);
            Mat imagemMatICAOWithoutBackground = await faceCropper.CropFace(imagemMatWithoutBackground, _utils.To2D(keypoints));
            #endregion

            #region "Passo 5 - Remoção do fundo 5.2"
            if (iCAONetParams.RemoveBackground)
            {
                evaluationResult.ICAOImageWithoutBackground = imagemMatICAOWithoutBackground.ToBytes();
            }
            #endregion

            #region "Passo 6"
            using Mat faceCroppedIcaoFormat = await faceCropper.CropFace(originalImage, _utils.To2D(keypoints));
            evaluationResult.ICAOImage = faceCroppedIcaoFormat.ToBytes();
            #endregion

            #region "Passo 7 - Olhos abertos"
            EyesOpenEstimator eyesOpenEstimator = new EyesOpenEstimator();
            evaluationResult.IsEyesOpened = await eyesOpenEstimator.Compute(keypointsForEvaluation, matFaceCroppedForEvaluation.Width, matFaceCroppedForEvaluation.Height);
            #endregion

            #region "Passo 8 - Óculos"
            GlassDetector glassDetector = new GlassDetector();
            evaluationResult.IsGlassCompliance = await glassDetector.Predict(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 9 - Emotion (Sorriso, principalmente)"
            EmotionDetector emotionDetector = new EmotionDetector();
            evaluationResult.IsSmilingCompliance = await emotionDetector.Predict(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 10 - Head Cover"
            HeadCoverDetector headCoverDetector = new HeadCoverDetector();
            evaluationResult.IsHeadCoverCompliance = await headCoverDetector.Predict(imagemMatICAOWithoutBackground);
            #endregion

            #region "Passo 11 - Boca Fechada"
            ClosedMouthDetector closedMouthDetector = new ClosedMouthDetector();
            evaluationResult.IsMouthClosed = await closedMouthDetector.Evaluate(_utils.To2D(keypointsForEvaluation));
            #endregion

            #region "Passo 12 - Olhos(s) Vemelho(s)"
            RedEyesDetector redEyesDetector = new RedEyesDetector();
            evaluationResult.IsRedEyeCompliance = await redEyesDetector.Predict(matFaceCroppedForEvaluation, _utils.To2D(keypointsForEvaluation));
            #endregion

            #region "Passo 13 - Maquiagem"
            MakeUpDetector makeUpDetector = new MakeUpDetector();
            evaluationResult.IsMakeUpCompliance = await makeUpDetector.Predict(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 14 - Eye Gaze Estimation"
            EyeGazeEstimator eyeGazeEstimator = new EyeGazeEstimator();
            evaluationResult.IsEyeGazeCompliance = await eyeGazeEstimator.Predict(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 15 - Head Pose"
            HeadPoseChecker headPoseChecker = new HeadPoseChecker();
            evaluationResult.IsHeadPoseCompliance = await headPoseChecker.Predict(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 17 - IsImagePixelated"
            evaluationResult.IsImagePixelatedComplicance = await _qualityChecker.IsImagePixelated(imagemMatICAOWithoutBackground);
            #endregion

            #region "Passo 18 - IsImagePosterized"
            evaluationResult.IsImagePosterizedCompliance = await _qualityChecker.IsImagePosterized(matFaceCroppedForEvaluation);
            #endregion

            #region "Passo 19 - IsImageOutOfFocus"
            evaluationResult.IsFocusCompliance = await _qualityChecker.IsImageOutOfFocus(imagemMatICAOWithoutBackground);
            #endregion

            #region "Passos - 20 (Assimetria), 21 (Uniformidade das Cores no Rosto), 22 (Sombra no Rosto), 23 (Reflexo de Luz no Rosto)"
            var tempQualityChecker = await _qualityChecker.IsImageUniformLightning(imagemMatWithoutBackground, _utils.To2D(keypointsForEvaluation));
            var temp = tempQualityChecker.Item2.Split(";");

            foreach (var item in temp)
            {
                if (item.ToLower().Contains("assimetria"))
                {
                    evaluationResult.IsLightningAsymmetryCompliance = (false, $"{EnumICAOResult.NOT_PASSED}. {item}");
                }
                else if (item.ToLower().Contains("uniformidade"))
                {
                    evaluationResult.IsColorUniformityCompliance = (false, $"{EnumICAOResult.NOT_PASSED}. {item}");
                }
                else if (item.ToLower().Contains("sombras"))
                {
                    evaluationResult.IsShadowCompliance = (false, $"{EnumICAOResult.NOT_PASSED}. {item}");
                }
                else if (item.ToLower().Contains("reflexos"))
                {
                    evaluationResult.IsLightReflextionCompliance = (false, $"{EnumICAOResult.NOT_PASSED}. {item}");
                }
            }
            foreach (var item in temp)
            {
                if (!item.ToLower().Contains("assimetria"))
                {
                    evaluationResult.IsLightningAsymmetryCompliance = (true, $"{EnumICAOResult.PASSED}.");
                }
                if (!item.ToLower().Contains("uniformidade"))
                {
                    evaluationResult.IsColorUniformityCompliance = (true, $"{EnumICAOResult.PASSED}.");
                }
                if (!item.ToLower().Contains("sombras"))
                {
                    evaluationResult.IsShadowCompliance = (true, $"{EnumICAOResult.PASSED}.");
                }
                if (!item.ToLower().Contains("reflexos"))
                {
                    evaluationResult.IsLightReflextionCompliance = (true, $"{EnumICAOResult.PASSED}.");
                }
            }
            #endregion

            return evaluationResult;
        }
    }
}
