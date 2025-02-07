#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class CahnHilliardImageInpainting
 * @brief Implements the Cahn-Hilliard image inpainting algorithm.
 * @see https://arxiv.org/abs/2412.11946
 */
class CahnHilliardImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit CahnHilliardImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~CahnHilliardImageInpainting() {}

    public:
        void inpaint() override;
        void initParameters() override;

    private:
        void cahnHilliardInpainting(cv::Mat& c, const cv::Mat& mask, int nIters = 50,
                                    double D = 1.5, double gamma = 1.0, double dt = 0.01,
                                    double deltaX = 1.0, double deltaY = 1.0, bool useExplicitLaplacian = true);
};

