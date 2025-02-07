#include "datamanager.h"

#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <QDebug>

DataManager::DataManager() {}

bool DataManager::importImageFromFile(const std::string& filename)
{
    mImage = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (mImage.empty()) return false;

    mImageType = mImage.type();

    showImageType(mImage);

    mOriginalImage = mImage.clone();
    mNoisyImage = mImage.clone();
    mInpaintedImage = mImage.clone();

    mInpaintingMask = cv::Mat(mImage.size(), CV_8UC1, cv::Scalar(255));

    return true;
}

bool DataManager::exportImageToFile(const std::string& filename)
{
    if (mImage.empty()) return false;

    cv::Mat outputImage = mImage.clone();
    if (mImageType == CV_8UC1)
    {
        if (outputImage.channels() == 3)
            cv::cvtColor(outputImage, outputImage, cv::COLOR_BGR2GRAY);
    }
    else if (mImageType == CV_8UC3)
    {
        if (outputImage.channels() == 1)
            cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
    }

    if (mImageType == CV_64F)
    {
        if (outputImage.depth() == CV_64F)
            outputImage.convertTo(outputImage, CV_8U, 255.0);
    }
    else if (mImageType == CV_8U)
    {
        if (outputImage.depth() == CV_64F)
            outputImage.convertTo(outputImage, CV_64F, 1.0 / 255.0);
    }

    return cv::imwrite(filename, outputImage);
}

void DataManager::clearImage()
{
    mImage.release();
    mOriginalImage.release();
    mNoisyImage.release();
    mInpaintedImage.release();
    mInpaintingMask.release();
}

void DataManager::showImageType(const cv::Mat& image)
{
    int channels = image.channels();
    int depth = image.depth();

    std::string depthStr;
    int bitDepth = 0;
    switch (depth)
    {
        case CV_8U:
            depthStr = "8-bit";
            break;
        case CV_16U:
            depthStr = "16-bit";
            break;
        case CV_32F:
            depthStr = "32-bit float";
            break;
        default:
            depthStr = "Unknown Depth";
            break;
    }

    std::string channelsStr;
    std::string colorStr;
    if (channels == 1)
    {
        channelsStr = "1 channel";
        colorStr = "Grayscale";
    }
    else if (channels == 3)
    {
        channelsStr = "3 channels";
        colorStr = "RGB";
    }
    else if (channels == 4)
    {
        channelsStr = "4 channels";
        colorStr = "BGRA";
    }
    else
    {
        channelsStr = std::to_string(channels) + " channels";
        colorStr = "Unknown";
    }

    std::string statusMessage = "The image is " + depthStr + " with " + channelsStr + " (" + colorStr + ")";

    emit statusShowMessage(QString::fromStdString(statusMessage));
}
