#pragma once

#include "ImageInpaintingBase.h"

#include <opencv4/opencv2/core.hpp>

/**
 * @class BurgersViscousImageInpainting
 * @brief Implements vicsous Burguers' equation image inpainting algorithm.
 * @see https://arxiv.org/abs/2412.11946
 */
class BurgersViscousImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit BurgersViscousImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~BurgersViscousImageInpainting() {}

    public:
        void inpaint();
        void initParameters();

    private:
        void burgersViscousEquationInpainting(cv::Mat& u, const cv::Mat& mask, double nu, int nIters = 1000, double dt = 0.01, double dx = 0.1, double dy = 0.1, bool useUpwind = true);
};

