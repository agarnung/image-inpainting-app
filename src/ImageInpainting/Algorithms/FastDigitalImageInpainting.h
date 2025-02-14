#pragma once

#include "ImageInpaintingBase.h"

#include <opencv4/opencv2/core.hpp>

/**
 * @class FastDigitalImageInpainting
 * @brief Implementation of Fast Digital Image Inpainting described in
 *        M. M. Oliveira, B. Bowen, R. McKenna, Y.-S. Chang: Fast Digital Image Inpainting,
 *        Proc. of Int. Conf. on Visualization, Imaging and Image Processing (VIIP), pp. 261-266, 2001.
 * @see   Based on the implementation by in https://github.com/Mugichoko445/Fast-Digital-Image-Inpainting
 */
class FastDigitalImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit FastDigitalImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~FastDigitalImageInpainting() {}

    public:
        void inpaint() override;
        void initParameters() override;

    private:
        void fastInpaint(const cv::Mat& src, const cv::Mat& mask, const cv::Mat& kernel, cv::Mat& dst, int maxNumOfIter = 100);
};

