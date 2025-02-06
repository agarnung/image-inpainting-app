#pragma once

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <QPixmap>

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

        cv::Mat getImage() const { return mImage; }
        cv::Mat getNoisyImage() const { return mNoisyImage; }
        cv::Mat getOriginalImage() const { return mOriginalImage; }
        cv::Mat getInpaintedImage() const { return mInpaintedImage; }
        QPixmap getImagePixmap() const { return matToPixmap(mImage); }
        QPixmap getNoisyImagePixmap() const { return matToPixmap(mNoisyImage); }
        QPixmap getOriginalImagePixmap() const { return matToPixmap(mOriginalImage); }
        QPixmap getInpaintedImagePixmap() const { return matToPixmap(mInpaintedImage); }
        void setImage(const cv::Mat& image) { mImage = image; }
        void setNoisyImage(const cv::Mat& noisyImage) { mNoisyImage = noisyImage; }
        void setOriginalImage(const cv::Mat& originalImage) { mOriginalImage = originalImage; }
        void setInpaintedImage(const cv::Mat& inpaintedImage) { mInpaintedImage = inpaintedImage; }

        void imageToNoisyImage() { mImage = mNoisyImage; }
        void imageToOriginalImage() { mImage = mOriginalImage; }
        void imageToInpaintedImage() { mImage = mInpaintedImage; }
        void clearImage();

        static QPixmap matToPixmap(const cv::Mat& mat);

    private:
        cv::Mat mImage;
        cv::Mat mNoisyImage;
        cv::Mat mOriginalImage;
        cv::Mat mInpaintedImage;
};
