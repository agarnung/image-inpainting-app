#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class MaxwellHeavisideImageInpainting
 * @brief Implements the Maxwell-Heaviside image inpainting algorithm.
 * @see https://arxiv.org/abs/2412.11946
 */
class MaxwellHeavisideImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit MaxwellHeavisideImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~MaxwellHeavisideImageInpainting() {}

    public:
        void inpaint();
        void initParameters();

    private:
        void maxwellHeavisidePDEInpainting(cv::Mat& u, const cv::Mat& mask, int nIters = 1000, double c_wave = 3.0e8,
                                           double dt = 0.1, double alpha = 0.0, double beta = 0.0, double gamma = 1.0,
                                           bool useEquationOne = true, double epsilon_0 = 1.0, double mu_0 = 1.0,
                                           bool useEulerMethod = true, bool stationaryFields = false);
};

