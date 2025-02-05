#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class MaxwellHeavisideInpainting
 * @brief Implements the Maxwell-Heaviside inpainting algorithm.
 * @see https://arxiv.org/abs/2412.11946
 */
class MaxwellHeavisideInpainting : public ImageInpaintingBase
{
    public:
        explicit MaxwellHeavisideInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~MaxwellHeavisideInpainting() {}

    public:
        void inpaint();
        void initParameters();
};

