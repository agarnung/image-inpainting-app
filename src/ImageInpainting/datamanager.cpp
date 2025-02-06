#include "datamanager.h"

#include <opencv4/opencv2/imgcodecs.hpp>

DataManager::DataManager() {}

bool DataManager::importImageFromFile(const std::string& filename)
{
    mImage = cv::imread(filename, cv::IMREAD_UNCHANGED);

    if (mImage.empty())
        return false;

    mOriginalImage = mImage;
    mNoisyImage = mImage;
    mInpaintedImage = mImage;

    return true;
}

bool DataManager::exportImageToFile(const std::string& filename)
{
    if (mImage.empty())
        return false;

    return cv::imwrite(filename, mImage);
}

void DataManager::clearImage()
{
    cv::Mat newImage;
    setImage(newImage);
    setOriginalImage(newImage);
    setNoisyImage(newImage);
    setInpaintedImage(newImage);
}

QPixmap DataManager::matToPixmap(const cv::Mat& mat)
{
    if (mat.empty()) return QPixmap();

    cv::Mat matRGB;
    if (mat.channels() == 3) cv::cvtColor(mat, matRGB, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4) cv::cvtColor(mat, matRGB, cv::COLOR_BGRA2RGBA);
    else matRGB = mat.clone();

    QImage img(matRGB.data, matRGB.cols, matRGB.rows, matRGB.step, QImage::Format_RGB888);

    return QPixmap::fromImage(img);
}
