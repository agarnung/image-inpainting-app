#pragma once

#include "ImageInpaintingBase.h"

/**
 * @class TeleaImageInpainting
 * @brief Executes OpenCV's TELEA image inpainting algorithm
 */
class TeleaImageInpainting : public ImageInpaintingBase
{
    public:
        explicit TeleaImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~TeleaImageInpainting() {}

    public:
        void inpaint();
        void initParameters();
};

