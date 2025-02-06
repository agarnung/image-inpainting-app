#include "datamanager.h"

#include <opencv4/opencv2/imgcodecs.hpp>

DataManager::DataManager() {}

bool DataManager::importImageFromFile(const std::string& filename)
{
    mImage = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (mImage.empty()) return false;

    mOriginalImage = mImage.clone();
    mNoisyImage = mImage.clone();
    mInpaintedImage = mImage.clone();

    return true;
}

bool DataManager::exportImageToFile(const std::string& filename)
{
    if (mImage.empty()) return false;

    cv::Mat outputImage;
    cv::cvtColor(mImage, outputImage, cv::COLOR_RGB2BGR);
    return cv::imwrite(filename, outputImage);
}

void DataManager::clearImage()
{
    mImage.release();
    mOriginalImage.release();
    mNoisyImage.release();
    mInpaintedImage.release();
    // mInpaintingMask.release();
}

QPixmap DataManager::matToPixmap(const cv::Mat& mat)
{
    if (mat.empty()) return QPixmap();

    cv::Mat matRGB;
    if (mat.channels() == 3) cv::cvtColor(mat, matRGB, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4) cv::cvtColor(mat, matRGB, cv::COLOR_BGRA2RGBA);
    else matRGB = mat.clone();

    QImage img(matRGB.data, matRGB.cols, matRGB.rows, matRGB.step,
               matRGB.channels() == 4 ? QImage::Format_RGBA8888 :
               matRGB.channels() == 3 ? QImage::Format_RGB888 :
               matRGB.channels() == 1 ? QImage::Format_Grayscale8 : QImage::Format_Invalid);

    return QPixmap::fromImage(img);
}
