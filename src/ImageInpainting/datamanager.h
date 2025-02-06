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

        enum ViewMode { Original, Noisy, Inpainted, Mask };

        bool importImageFromFile(const std::string& filename);
        bool exportImageToFile(const std::string& filename);

        ViewMode getCurrentViewMode() const { return mCurrentViewMode; }
        void setCurrentViewMode(ViewMode mode) { mCurrentViewMode = mode; }

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
        inline void setMask(const QPixmap& pixmap) { mInpaintingMask = pixmapToMat(pixmap); }

        inline void imageToNoisyImage() { mImage = mNoisyImage; setCurrentViewMode(ViewMode::Noisy); }
        inline void imageToOriginalImage() { mImage = mOriginalImage; setCurrentViewMode(ViewMode::Original); }
        inline void imageToInpaintedImage() { mImage = mInpaintedImage; setCurrentViewMode(ViewMode::Inpainted); }
        inline void imageToMask() { mImage = mInpaintingMask; setCurrentViewMode(ViewMode::Mask); }

        void clearImage();

        static QPixmap matToPixmap(const cv::Mat& mat);
        static cv::Mat pixmapToMat(const QPixmap& pixmap);

    private:
        cv::Mat mImage;
        cv::Mat mNoisyImage;
        cv::Mat mOriginalImage;
        cv::Mat mInpaintedImage;
        cv::Mat mInpaintingMask;

        ViewMode mCurrentViewMode = Original;
};
