#pragma once

#include <QPixmap>
#include <QImage>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#define DELETE(x) if(x) { delete x; x = nullptr; }

double MSE(const cv::Mat& img1, const cv::Mat& img2);
void MSE(const cv::Mat& I, const cv::Mat& u, double& mse, double& rmse);
double PSNR(const cv::Mat& img1, const cv::Mat& img2);
double SSIM(const cv::Mat& img1, const cv::Mat& img2);

QPixmap matToPixmap(const cv::Mat& mat);
cv::Mat pixmapToMat(const QPixmap &pixmap);
cv::Mat universalConvertTo(const cv::Mat& src, int outputType);
