#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class HarmonicImageInpainting
 * @brief The heat equation (harmonic inpainting)
 * @see https://es.mathworks.com/matlabcentral/fileexchange/34356-higher-order-total-variation-inpainting?status=SUCCESS
 */
class HarmonicImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit HarmonicImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~HarmonicImageInpainting() {}

        void inpaint() override;
        void initParameters() override;

    private:
        /**
         * heat_equation.m: the heat equation (harmonic inpainting)
         *
         * @param [in] g cv::Mat Imagen de partida, corrupta
         * @param [in, out] u cv::Mat Imagen final del inpainting
         * @param nIters int Número de iteraciones del algoritmo
         * @param dt double Paso de tiempo
         * @param alpha double Difusividad constante
         * @param lambda0 double Regularizador. Es cero en la regiones perdidas, pero constante en el resto de la imagen,
         *        pues su contorno se trata como borde de máxima magnitud de gradiente
         * @see https://es.mathworks.com/matlabcentral/fileexchange/34356-higher-order-total-variation-inpainting?status=SUCCESS
         *
         * @arg useMatrixForm bool true == aproximación con cv::Laplacian, algo más rápido pero menos conservador
         *
         * @note Se usa un esquema de paso de tiempo explícito
         */
        void heatEquationHarmonicInpainting(const cv::Mat& g, cv::Mat& u, const cv::Mat& lambda, int nIters, double dt, double alpha, bool useMatrixForm = false);
};

