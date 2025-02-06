#pragma once

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#define DELETE(x) if(x) { delete x; x = nullptr; }

// MSE = (1/N) * Σ (X[i] - Y[i])^2
static void MSE(const cv::Mat& I, const cv::Mat& u, double& mse, double& rmse)
{
    cv::Mat error;
    cv::subtract(I, u, error); // I - u
    cv::Mat error_sq = error.mul(error); // (I - u)^2

    mse = cv::sum(error_sq).val[0] / I.total();
    rmse = std::sqrt(mse);
}

// MSE = (1/N) * Σ (X[i] - Y[i])^2
static double MSE(cv::Mat const& img1, cv::Mat const& img2)
{
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    int npixels = img1.size().area() * img1.channels();

    cv::Mat errorImage = img1 - img2;
    cv::multiply(errorImage, errorImage, errorImage);

    return (double)cv::norm(errorImage, cv::NORM_L1) / npixels;
}

// PSNR = 20 * log10(Imax) - 10 * log10(MSE)
static double PSNR(cv::Mat const &img1, cv::Mat const &img2)
{
    double mse, rmse;
    MSE(img1, img2, mse, rmse);

    // Calcular Imax como el valor máximo de píxel en la imagen
    double maxVal1, maxVal2;
    cv::minMaxLoc(img1, nullptr, &maxVal1);
    cv::minMaxLoc(img2, nullptr, &maxVal2);
    double Imax = (maxVal1 > maxVal2) ? maxVal1 : maxVal2;

    return (double)20.0 * std::log10(Imax) - 10.0 * std::log10(mse);
}

/// @overload
static double SSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
    CV_Assert(img1.type() == CV_64F);

    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;

    cv::Mat img1_sq, img2_sq, img1_img2;
    cv::multiply(img1, img1, img1_sq);
    cv::multiply(img2, img2, img2_sq);
    cv::multiply(img1, img2, img1_img2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq, mu2_sq, mu1_mu2;
    cv::multiply(mu1, mu1, mu1_sq);
    cv::multiply(mu2, mu2, mu2_sq);
    cv::multiply(mu1, mu2, mu1_mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1_sq, sigma1_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_sq, sigma2_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1_img2, sigma12, cv::Size(11, 11), 1.5);

    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    cv::Mat t4, t5;
    t4 = mu1_sq + mu2_sq + C1;
    t5 = sigma1_sq + sigma2_sq + C2;
    cv::Mat t6 = t4.mul(t5);

    cv::Mat ssim_map;
    cv::divide(t3, t6, ssim_map);

    return cv::mean(ssim_map)[0];
}
