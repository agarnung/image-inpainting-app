#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class NavierStokesImageInpainting
 * @brief Executes OpenCV's NS image inpainting algorithm
 */
class NavierStokesImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit NavierStokesImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~NavierStokesImageInpainting() {}

    public:
        void inpaint();
        void initParameters();
};

