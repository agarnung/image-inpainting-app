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
        ~DataManager() {};

        bool importImageFromFile(const std::string& filename);
        bool exportImageToFile(const std::string& filename);

        inline cv::Mat getImage() const { return mImage; }
        inline cv::Mat getNoisyImage() const { return mNoisyImage; }
        inline cv::Mat getOriginalImage() const { return mOriginalImage; }
        inline cv::Mat getInpaintedImage() const { return mInpaintedImage; }
        inline cv::Mat getMask() const { return mInpaintingMask; }
        inline QPixmap getImagePixmap() const { return matToPixmap(mImage); }
        inline QPixmap getNoisyImagePixmap() const { return matToPixmap(mNoisyImage); }
        inline QPixmap getOriginalImagePixmap() const { return matToPixmap(mOriginalImage); }
        inline QPixmap getInpaintedImagePixmap() const { return matToPixmap(mInpaintedImage); }
        inline QPixmap getMaskPixmap() const { return matToPixmap(mInpaintingMask); }
        inline void setImage(const cv::Mat& image) { mImage = image; }
        inline void setNoisyImage(const cv::Mat& noisyImage) { mNoisyImage = noisyImage; }
        inline void setOriginalImage(const cv::Mat& originalImage) { mOriginalImage = originalImage; }
        inline void setInpaintedImage(const cv::Mat& inpaintedImage) { mInpaintedImage = inpaintedImage; }
        inline void setMask(const cv::Mat& mask) { mInpaintingMask = mask; }

        inline void imageToNoisyImage() { mImage = mNoisyImage; }
        inline void imageToOriginalImage() { mImage = mOriginalImage; }
        inline void imageToInpaintedImage() { mImage = mInpaintedImage; }
        inline void imageToMask() { mImage = mInpaintingMask; }

        void clearImage();

        static QPixmap matToPixmap(const cv::Mat& mat);

    private:
        cv::Mat mImage;
        cv::Mat mNoisyImage;
        cv::Mat mOriginalImage;
        cv::Mat mInpaintedImage;
        cv::Mat mInpaintingMask;
};
