#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class MaxwellHeavisideImageInpainting
 * @brief Implements the Maxwell-Heaviside image inpainting algorithm.
 * @see https://arxiv.org/abs/2412.11946
 */
class MaxwellHeavisideImageInpainting : public ImageInpaintingBase
{
    public:
        explicit MaxwellHeavisideImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~MaxwellHeavisideImageInpainting() {}

    public:
        void inpaint();
        void initParameters();
};

