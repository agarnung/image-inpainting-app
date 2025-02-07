#include "utils.h"

#include <opencv4/opencv2/highgui.hpp>

void MSE(const cv::Mat& I, const cv::Mat& u, double& mse, double& rmse)
{
    cv::Mat error;
    cv::subtract(I, u, error); // I - u
    cv::Mat error_sq = error.mul(error); // (I - u)^2

    mse = cv::sum(error_sq).val[0] / I.total();
    rmse = std::sqrt(mse);
}

double MSE(const cv::Mat& img1, const cv::Mat& img2)
{
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    int npixels = img1.size().area() * img1.channels();

    cv::Mat errorImage = img1 - img2;
    cv::multiply(errorImage, errorImage, errorImage);

    return (double)cv::norm(errorImage, cv::NORM_L1) / npixels;
}

double PSNR(const cv::Mat& img1, const cv::Mat& img2)
{
    double mse, rmse;
    MSE(img1, img2, mse, rmse);

    double maxVal1, maxVal2;
    cv::minMaxLoc(img1, nullptr, &maxVal1);
    cv::minMaxLoc(img2, nullptr, &maxVal2);
    double Imax = std::max(maxVal1, maxVal2);

    return 20.0 * std::log10(Imax) - 10.0 * std::log10(mse);
}

double SSIM(const cv::Mat& img1, const cv::Mat& img2)
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

QPixmap matToPixmap(const cv::Mat& mat)
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

cv::Mat pixmapToMat(const QPixmap &pixmap)
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
