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

    mInpaintingMask = cv::Mat(mImage.size(), CV_8UC1, cv::Scalar(255));

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
    mInpaintingMask.release();
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

cv::Mat DataManager::pixmapToMat(const QPixmap &pixmap)
{
    QImage image = pixmap.toImage();

    if (image.isNull())
        return cv::Mat();

    cv::Mat mat;

    if (image.format() == QImage::Format_RGB888)
    {
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    }
    else if (image.format() == QImage::Format_RGBA8888)
    {
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.bits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        cv::Mat grayMat;
        cv::cvtColor(mat, grayMat, cv::COLOR_BGRA2GRAY);
        return grayMat;
    }
    else if (image.format() == QImage::Format_Grayscale8)
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.bits(), image.bytesPerLine());
    else if (image.format() == QImage::Format_RGB32)
    {
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.bits(), image.bytesPerLine());

        cv::Mat grayMat(image.height(), image.width(), CV_8UC1);

        for (int y = 0; y < mat.rows; y++)
        {
            for (int x = 0; x < mat.cols; x++)
            {
                cv::Vec4b pixel = mat.at<cv::Vec4b>(y, x);
                uchar r = pixel[2];
                uchar g = pixel[1];
                uchar b = pixel[0];

                uchar gray = static_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);

                grayMat.at<uchar>(y, x) = gray;
            }
        }

        return grayMat;
    }
    else
        return cv::Mat();

    return mat.clone();
}

