#pragma once

#include <opencv4/opencv2/core.hpp>

/**
 * @class DataManager
 * @brief Handles the image data, including import, export, and transformations.
 *
 * Manages different versions of the image: original, corrupted (original by default), and restored.
 */
class DataManager
{
    public:
        DataManager();

        cv::Mat getImage() const {return mImage;}
        cv::Mat getNoisyImage() const {return mNoisyImage;}
        cv::Mat getOriginalImage() const {return mOriginalImage;}
        cv::Mat getInpaintedImage() const {return mInpaintedImage;}

    private:
        cv::Mat mImage;
        cv::Mat mNoisyImage;
        cv::Mat mOriginalImage;
        cv::Mat mInpaintedImage;
};
