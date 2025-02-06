/**
 * Comunión de funciones útiles y frecuentes en procesamiento de imagen y visión por computador
 * Índice:
 *  - Métricas de calidad
 *  - Cálculo diferencial
 *  - Cálculo variacional
 *  - Deburring
 *  - Denoising
 *  - Fourier
 *  - Visualización
 *  - 3D
 */

#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

namespace cvlib {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                             Métricas de calidad ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MSE = (1/N) * Σ (X[i] - Y[i])^2
double MSE(cv::Mat const& img1, cv::Mat const& img2)
{
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    int npixels = img1.size().area() * img1.channels();

    cv::Mat errorImage = img1 - img2;
    cv::multiply(errorImage, errorImage, errorImage);

    return (double)cv::norm(errorImage, cv::NORM_L1) / npixels;
}

// MSE = (1/N) * Σ (X[i] - Y[i])^2
void MSE(const cv::Mat& I, const cv::Mat& u, double& mse, double& rmse)
{
    cv::Mat error;
    cv::subtract(I, u, error); // I - u
    cv::Mat error_sq = error.mul(error); // (I - u)^2

    mse = cv::sum(error_sq).val[0] / I.total();
    rmse = std::sqrt(mse);
}

// PSNR = 20 * log10(Imax) - 10 * log10(MSE)
double PSNR(cv::Mat const &img1, cv::Mat const &img2)
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

// SNR = 20 * log10(Potencia de la señal / MSE) = 20 * log10((1/N * sum(i^2)) / MSE)
double SNR(cv::Mat const &img1, cv::Mat const &img2)
{
    double mse, rmse;
    MSE(img1, img2, mse, rmse);

    double signalPower = cv::norm(img2, cv::NORM_L2);
    signalPower *= signalPower;
    int npixels = img2.size().area() * img2.channels();

    signalPower /= npixels;

    return 20.0 * std::log10(signalPower / mse);
}

// SSIM = ((2 * μ_x * μ_y + C1) * (2 * σ_xy + C2)) / ((μ_x^2 + μ_y^2 + C1) * (σ_x^2 + σ_y^2 + C2))
void SSIM(cv::Mat const &img1, cv::Mat const &img2, cv::Mat &ssim_map, float L = 255.0f, float k1 = 0.01f, float k2 = 0.03f, int size = 11)
{
    float const C1 = k1 * k1 * L * L;
    float const C2 = k2 * k2 * L * L;

    cv::Mat img1_sq, img2_sq, img1_img2;
    cv::multiply(img1, img1, img1_sq);
    cv::multiply(img2, img2, img2_sq);
    cv::multiply(img1, img2, img1_img2);

    cv::Mat mu_x, mu_y;
    cv::boxFilter(img1, mu_x, CV_32F, cv::Size(size, size));
    cv::boxFilter(img2, mu_y, CV_32F, cv::Size(size, size));

    cv::Mat mu_x_sq, mu_y_sq, mu_xy;
    cv::multiply(mu_x, mu_x, mu_x_sq);
    cv::multiply(mu_y, mu_y, mu_y_sq);
    cv::multiply(mu_x, mu_y, mu_xy);

    cv::Mat sigma_x_sq, sigma_y_sq, sigma_xy;
    cv::boxFilter(img1_sq, sigma_x_sq, CV_32F, cv::Size(size, size));
    cv::boxFilter(img2_sq, sigma_y_sq, CV_32F, cv::Size(size, size));
    cv::boxFilter(img1_img2, sigma_xy, CV_32F, cv::Size(size, size));
    sigma_x_sq -= mu_x_sq;
    sigma_y_sq -= mu_y_sq;
    sigma_xy -= mu_xy;

    cv::Mat A, B, C, D;

    cv::multiply(mu_x, mu_y, A);
    A *= 2.0;
    A += cv::Scalar::all(C1);

    B = 2.0 * sigma_xy;
    B += C2;

    cv::multiply(A, B, ssim_map);

    C = sigma_x_sq + sigma_y_sq;
    C += cv::Scalar::all(C1);

    D = sigma_x_sq + sigma_y_sq;
    D *= C2;

    cv::multiply(C, D, C);

    cv::divide(ssim_map, C, ssim_map);
}

/// @overload
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

/**
 * @brief Implementation of the Universal Image Quality Index (QI).
 *
 * This is an efficient implementation of the algorithm for calculating the universal image quality
 * index proposed by Zhou Wang and Alan C. Bovik.
 *
 * Please refer to the paper "A Universal Image Quality Index" by Zhou Wang and Alan C. Bovik,
 * published in IEEE Signal Processing Letters, 2001.
 *
 * In order to run this function, you must have OpenCV installed.
 *
 * @note The input images must be of the same size.
 *
 * @param img1 The original image (reference).
 * @param img2 The test image to compare against the original.
 * @param block_size The size of the block used in the computation of the quality map (default is 8x8).
 *
 * @return A tuple containing:
 *   - qi: A scalar value representing the overall quality index of the test image, with a range of [-1, 1].
 *   - qi_map: A matrix representing the quality map of the test image. The size of the map is
 *             (img_size - block_size + 1).
 *
 * @usage
 * Load the original and test images into two matrices (e.g., img1 and img2).
 * You can call this function in one of the following ways:
 *
 * 1. Default block size (suggested):
 *    @code
 *    auto [qi, qi_map] = img_qi(img1, img2);
 *    @endcode
 *
 * 2. Custom block size:
 *    @code
 *    auto [qi, qi_map] = img_qi(img1, img2, block_size);
 *    @endcode
 *
 * Example output:
 *
 * @code
 * std::cout << "Quality Index: " << qi << std::endl;
 * cv::imshow("Quality Map", (qi_map + 1) / 2); // Show the quality map as an image
 * @endcode
 *
 * @see "A Universal Image Quality Index", Zhou Wang, Alan C. Bovik, IEEE Signal Processing Letters, 2001.
 */
void img_qi(const cv::Mat& img1, const cv::Mat& img2, int block_size, double& quality, cv::Mat& quality_map) {
    if (img1.empty() || img2.empty() || img1.size() != img2.size())
    {
        quality = -std::numeric_limits<double>::infinity();
        quality_map = cv::Mat::ones(img1.size(), CV_64F) * -1;
        return;
    }

    int N = block_size * block_size;
    cv::Mat sum2_filter = cv::Mat::ones(block_size, block_size, CV_64F);

    cv::Mat img1_sq, img2_sq, img12;
    cv::multiply(img1, img1, img1_sq);
    cv::multiply(img2, img2, img2_sq);
    cv::multiply(img1, img2, img12);

    cv::Mat img1_sum, img2_sum, img1_sq_sum, img2_sq_sum, img12_sum;

    cv::filter2D(img1, img1_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(img2, img2_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(img1_sq, img1_sq_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(img2_sq, img2_sq_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(img12, img12_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat img12_sum_mul = img1_sum.mul(img2_sum);
    cv::Mat img12_sq_sum_mul = img1_sum.mul(img1_sum) + img2_sum.mul(img2_sum);

    cv::Mat numerator = 4 * (N * img12_sum - img12_sum_mul).mul(img12_sum_mul);
    cv::Mat denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul;
    cv::Mat denominator = denominator1.mul(img12_sq_sum_mul);

    quality_map = cv::Mat::ones(denominator.size(), CV_64F);

    cv::Mat index1 = (denominator1 == 0) & (img12_sq_sum_mul != 0);
    quality_map.setTo(2 * img12_sum_mul / img12_sq_sum_mul, index1);

    cv::Mat index2 = (denominator != 0);
    quality_map.setTo(numerator / denominator, index2);

    cv::Scalar mean_quality = cv::mean(quality_map);
    quality = mean_quality[0];
}

/// @overload
double img_qi(const cv::Mat& img1, const cv::Mat& img2, int block_size)
{
    if (img1.empty() || img2.empty() || img1.size() != img2.size())
        return -std::numeric_limits<double>::infinity();

    int N = block_size * block_size;

    cv::Mat img1_f, img2_f;
    img1.convertTo(img1_f, CV_64F);
    img2.convertTo(img2_f, CV_64F);

    cv::Mat sum2_filter = cv::Mat::ones(block_size, block_size, CV_64F);

    cv::Mat img1_sq, img2_sq, img12;
    cv::multiply(img1_f, img1_f, img1_sq);
    cv::multiply(img2_f, img2_f, img2_sq);
    cv::multiply(img1_f, img2_f, img12);

    cv::Mat img1_sum, img2_sum, img1_sq_sum, img2_sq_sum, img12_sum;

    cv::filter2D(img1_f, img1_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(img2_f, img2_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(img1_sq, img1_sq_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(img2_sq, img2_sq_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(img12, img12_sum, -1, sum2_filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    cv::Mat img12_sum_mul = img1_sum.mul(img2_sum);
    cv::Mat img12_sq_sum_mul = img1_sum.mul(img1_sum) + img2_sum.mul(img2_sum);

    cv::Mat numerator = 4 * (N * img12_sum - img12_sum_mul).mul(img12_sum_mul);
    cv::Mat denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul;
    cv::Mat denominator = denominator1.mul(img12_sq_sum_mul);

    cv::Mat index1 = (denominator1 == 0) & (img12_sq_sum_mul != 0);
    cv::Mat result1 = 2 * img12_sum_mul / img12_sq_sum_mul;

    cv::Mat index2 = (denominator != 0);
    cv::Mat result2 = numerator / denominator;

    index1.convertTo(index1, CV_64F);
    index2.convertTo(index2, CV_64F);

    cv::Scalar mean_quality = cv::mean(result1.mul(index1) + result2.mul(index2));

    return mean_quality[0];
}

double computeAverageGradient(const cv::Mat& img)
{
    if (img.empty())
        return 0.0;

    cv::Mat img_f;
    img.convertTo(img_f, CV_64F);

    cv::Mat grad_x, grad_y;
    cv::Sobel(img_f, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(img_f, grad_y, CV_64F, 0, 1, 3);

    cv::Mat grad_magnitude;
    cv::magnitude(grad_x, grad_y, grad_magnitude);

    cv::Scalar mean_grad = cv::mean(grad_magnitude);

    return mean_grad[0];
}

// Function to compute PSNR using a different method
double PSNRKE(const cv::Mat& u_true, const cv::Mat& u) {
    CV_Assert(u_true.size() == u.size()); // Asegúrate de que las dimensiones coincidan
    int n = u.rows;
    int m = u.cols;

    // Encuentra el valor máximo de píxel en la imagen verdadera
    double M = *std::max_element(u_true.begin<double>(), u_true.end<double>()); // Máximo valor

    // Calcula las diferencias cuadradas
    cv::Mat diff;
    cv::absdiff(u_true, u, diff); // Calcula la diferencia absoluta
    diff = diff.mul(diff); // Cuadrar las diferencias

    // Sumar las diferencias cuadradas
    double sumSquaredDiffs = cv::sum(diff)[0];

    // Evitar la división por cero
    if (sumSquaredDiffs == 0) {
        return std::numeric_limits<double>::infinity(); // PSNR es infinito si no hay error
    }

    // Calcula el PSNR
    return 10 * log10((n * m * M * M) / sumSquaredDiffs);
}

// Function to compute the signal-to-noise ratio (SNR)
double snr(const cv::Mat& noisyData, const cv::Mat& original) {
    // Compute mean of original
    cv::Scalar meanOriginal = cv::mean(original);
    cv::Mat tmp = original - meanOriginal[0];

    // Compute variance of original
    double varOriginal = cv::sum(tmp.mul(tmp))[0];

    // Compute noise
    cv::Mat noise = noisyData - original;
    cv::Scalar meanNoise = cv::mean(noise);
    tmp = noise - meanNoise[0];

    // Compute variance of noise
    double varNoise = cv::sum(tmp.mul(tmp))[0];

    // Calculate SNR
    if (varNoise == 0) {
        return 999.99; // INF, clean image
    } else {
        return 10 * log10(varOriginal / varNoise);
    }
}

// Function to compute SNR using a different definition
double snrg(const cv::Mat& x, const cv::Mat& n) {
    return 20 * log10(cv::norm(x, cv::NORM_L2) / cv::norm(n, cv::NORM_L2));
}

// Function to compute PSNR
double psnr255(const cv::Mat& z, const cv::Mat& u) {
    CV_Assert(z.size() == u.size()); // Ensure sizes match
    double sum = 0.0;

    for (int i = 0; i < z.rows; i++) {
        for (int j = 0; j < z.cols; j++) {
            sum += pow(z.at<uchar>(i, j) - u.at<uchar>(i, j), 2);
        }
    }

    sum = sum / (z.rows * z.cols) + 0.0001; // Prevent division by zero
    return 10 * log10(pow(255, 2) / sum);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                             Cálculo diferencial ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Dx(i, j) = X(i, j + 1) - X(i, j), para i en [0, rows-1] y j en [0, cols-2].
void HorizontalGradientWithBackwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));

    int valuesPerRow = (X.cols - 1) * X.channels();

    for (int i = 0; i < X.rows; ++i)
    {
        const float *xjm1 = X.ptr<float>(i);
        const float *xj = xjm1 + X.channels();

        float *pdx = Dx.ptr<float>(i) + X.channels();

        for (int j = 0; j < valuesPerRow; ++j, ++xj, ++xjm1, ++pdx)
            *pdx = (*xj - *xjm1);
    }
}

/// Dy(i, j) = X(i, j) - X(i - 1, j), para i en [1, rows-1] y j en [0, cols-1].
void VerticalGradientWithBackwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));

    int valuesPerRow = X.channels() * X.cols;

    for (int i = 1; i < X.rows; ++i)
    {
        const float *xi = X.ptr<float>(i);
        const float *xim1 = X.ptr<float>(i - 1);

        float *pdy = Dx.ptr<float>(i);

        for (int j = 0; j < valuesPerRow; ++j, ++xi, ++xim1, ++pdy)
            *pdy = (*xi - *xim1);
    }
}

/// Dx(i, j) = X(i, j + 1) - X(i, j), para i en [0, rows-1] y j en [0, cols-1].
void HorizontalGradientWithForwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
    {
        return;
    }

    Dx.create(X.size(), CV_32FC1);
    Dx.setTo(cv::Scalar::all(0));

    for (int i = 0; i < X.rows; ++i)
    {
        const float *xj = X.ptr<float>(i);
        const float *xjp1 = xj + X.channels();

        float *pdx = Dx.ptr<float>(i);

        for (int j = 0; j < X.cols; ++j, ++xj, ++xjp1, ++pdx)
            *pdx = (*xjp1 - *xj);
    }
}

/// Dy(i, j) = X(i + 1, j) - X(i, j), para i en [0, rows-2] y j en [0, cols-1].
void VerticalGradientWithForwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dx.create(X.size(), CV_32FC1);
    Dx.setTo(cv::Scalar::all(0));

    int valuesPerRow = X.cols * X.channels();

    for (int i = 0; i < X.rows - 1; ++i)
    {
        const float *xi = X.ptr<float>(i);
        const float *xip1 = X.ptr<float>(i + 1);

        float *pdy = Dx.ptr<float>(i);

        for (int j = 0; j < X.cols; ++j, ++xi, ++xip1, ++pdy)
            *pdy = (*xip1 - *xi);
    }
}

/// Dx(i, j) = 0.5 * (X(i, j + 1) - X(i, j - 1)), para i en [0, rows-1] y j en [1, cols-2].
void HorizontalGradientWithCenteredScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));

    int valuesPerRow = (X.cols - 2) * X.channels();

    for (int i = 0; i < X.rows; ++i)
    {
        const float *xjm1 = X.ptr<float>(i);
        const float *xjp1 = xjm1 + 2 * X.channels();

        float *dx = Dx.ptr<float>(i) + X.channels();

        for (int j = 0; j < valuesPerRow; ++j, ++dx, ++xjm1, ++xjp1)
            *dx += (0.5 * ((*xjp1) - (*xjm1)));
    }
}

/// Dy(i, j) = 0.5 * (X(i + 1, j) - X(i - 1, j)), para i en [1, rows-2] y j en [0, cols-1].
void VerticalGradientWithCenteredScheme(cv::Mat const &X, cv::Mat &Dy)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dy.create(X.size(), CV_32F);
    Dy.setTo(cv::Scalar::all(0));

    int valuesPerRow = X.cols * X.channels();

    for (int i = 1; i < X.rows - 1; ++i)
    {
        const float *xim1 = X.ptr<float>(i - 1);
        const float *xip1 = X.ptr<float>(i + 1);

        float *dy = Dy.ptr<float>(i);

        for (int j = 0; j < valuesPerRow; ++j, ++xim1, ++xip1, ++dy)
            *dy = (0.5 * ((*xip1) - (*xim1)));
    }
}

/// Dx(i, j) = (1/12) * (-X(i, j - 2) + 8*X(i, j - 1) - 8*X(i, j + 1) + X(i, j + 2)), para i en [0, rows-1] y j en [2, cols-3].
void HorizontalGradientWith5PointsScheme(const cv::Mat &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));

    cv::Mat kernel = (cv::Mat_<float>(1, 5) << -1, 8, 0, -8, 1);
    kernel /= 12.0;
    cv::filter2D(X, Dx, Dx.depth(), kernel);
}

/// Dy(i, j) = (1/12) * (-X(i - 2, j) + 8*X(i - 1, j) - 8*X(i + 1, j) + X(i + 2, j)), para i en [2, rows-3] y j en [0, cols-1].
void VerticalGradientWith5PointsScheme(const cv::Mat &X, cv::Mat &Dx)
{
    if (!X.data || X.depth() != CV_32F)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));

    cv::Mat kernel = (cv::Mat_<float>(5, 1) << -1, 8, 0, -8, 1);
    kernel /= 12.0;
    cv::filter2D(X, Dx, Dx.depth(), kernel);
}


/// divX(i, j) = DX1(i, j) + DX2(i, j), donde DX1(i, j) es el gradiente horizontal y DX2(i, j) es el gradiente vertical.
void DivergenceWithBackwardScheme(cv::Mat const &X1, cv::Mat const &X2, cv::Mat &divX)
{
    if (!X1.data || !X2.data)
    {
        return;
    }

    divX.create(X1.size(), CV_32FC1);
    divX.setTo(cv::Scalar::all(0));

    cv::Mat DX1;
    HorizontalGradientWithBackwardScheme(X1, DX1);

    cv::Mat DX2;
    VerticalGradientWithBackwardScheme(X2, DX2);

    divX = DX1 + DX2;
}

/// divX(i, j) = DX1(i, j) + DX2(i, j), donde DX1(i, j) es el gradiente horizontal y DX2(i, j) es el gradiente vertical.
void DivergenceWithCenteredScheme(cv::Mat const &X1, cv::Mat const &X2, cv::Mat &divX)
{
    if (!X1.data || !X2.data)
    {
        return;
    }

    divX.create(X1.size(), CV_32FC1);
    divX.setTo(cv::Scalar::all(0));

    cv::Mat DX1;
    HorizontalGradientWithCenteredScheme(X1, DX1);

    cv::Mat DX2;
    VerticalGradientWithCenteredScheme(X2, DX2);

    divX = DX1 + DX2;
}

/// ∇²u(x, y) = ∂²u/∂x² + ∂²u/∂y²
cv::Mat LaplacianWithCenteredScheme(cv::Mat const& u)
{
    if (!u.data)
        return cv::Mat();

    cv::Mat laplacian;
    laplacian.create(u.size(), CV_32FC1);
    laplacian.setTo(cv::Scalar::all(0));

    cv::Mat DX1, DX2;
    HorizontalGradientWithCenteredScheme(u, DX1);
    VerticalGradientWithCenteredScheme(u, DX2);

    cv::Mat DDX1, DDX2;
    HorizontalGradientWithCenteredScheme(DX1, DDX1);
    VerticalGradientWithCenteredScheme(DX2, DDX2);

    laplacian = DDX1 + DDX2;

    return laplacian;
}

/// Dxd(i, j) = -div(Ux(i, j), Uy(i, j)), donde Ux y Uy son el gradiente suavizado de Xd.
void gradIsotropicTVSmoothed(cv::Mat const &Xd, cv::Mat &Dxd, float mu)
{
    if (!Xd.data)
        return;

    Dxd.create(Xd.size(), CV_32FC1);
    Dxd.setTo(cv::Scalar::all(0));

    cv::Mat Ux(Xd.size(), CV_32F, 1);
    cv::Mat Uy(Xd.size(), CV_32F, 1);

    HorizontalGradientWithForwardScheme(Xd, Ux);
    VerticalGradientWithForwardScheme(Xd, Uy);

    int valuesPerRow = Xd.cols * Xd.channels();

    for (int i = 0; i < Xd.rows; ++i)
    {
        float *uxptr = Ux.ptr<float>(i);
        float *uyptr = Uy.ptr<float>(i);

        for (int j = 0; j < valuesPerRow; ++j, ++uxptr, ++uyptr)
        {
            float norm_DX = MAX(hypotf(*uxptr, *uyptr), mu);
            *uxptr /= norm_DX;
            *uyptr /= norm_DX;
        }
    }

    Dxd.setTo(0);
    DivergenceWithBackwardScheme(Ux, Uy, Dxd);
    Dxd *= -1.0f;
}

/// Función para calcular derivadas forward y backward
void calculateForwardBackward(const cv::Mat& input, cv::Mat& forward, cv::Mat& backward, bool isX)
{
    int m = input.rows;
    int n = input.cols;

    int type = input.type();
    int cvType = (type == CV_32F) ? CV_32F : CV_64F;

    forward = cv::Mat::zeros(m, n, cvType);  // u(i,j+1) - u(i,j)
    backward = cv::Mat::zeros(m, n, cvType); // u(i,j) - u(i,j-1)

    if (isX)
    {
        forward.colRange(0, n - 1) = input.colRange(1, n) - input.colRange(0, n - 1);
        backward.colRange(1, n) = input.colRange(1, n) - input.colRange(0, n - 1);
    }
    else
    {
        forward.rowRange(0, m - 1) = input.rowRange(1, m) - input.rowRange(0, m - 1);
        backward.rowRange(1, m) = input.rowRange(1, m) - input.rowRange(0, m - 1);
    }
}

/// Function to compute Min-mod derivatives
cv::Mat minmodM(const cv::Mat& a, const cv::Mat& b) {
    auto sign = [](const cv::Mat& mat) {
        int matType = mat.type();

        cv::Mat signMat = cv::Mat::zeros(mat.size(), matType);

        if (matType == CV_32F) {
            for (int i = 0; i < mat.rows; ++i) {
                for (int j = 0; j < mat.cols; ++j) {
                    if (mat.at<float>(i, j) > 0) {
                        signMat.at<float>(i, j) = 1;
                    } else if (mat.at<float>(i, j) < 0) {
                        signMat.at<float>(i, j) = -1;
                    } else {
                        signMat.at<float>(i, j) = 0;
                    }
                }
            }
        } else if (matType == CV_64F) {
            for (int i = 0; i < mat.rows; ++i) {
                for (int j = 0; j < mat.cols; ++j) {
                    if (mat.at<double>(i, j) > 0) {
                        signMat.at<double>(i, j) = 1;
                    } else if (mat.at<double>(i, j) < 0) {
                        signMat.at<double>(i, j) = -1;
                    } else {
                        signMat.at<double>(i, j) = 0;
                    }
                }
            }
        }
        return signMat;
    };

    // Verificar el tipo de las matrices de entrada
    int aType = a.type();
    int bType = b.type();

    if (aType != bType) {
        CV_Error(cv::Error::StsBadArg, "minmodM - Las matrices de entrada deben tener el mismo tipo");
    }

    // Crear matrices para los valores absolutos con el mismo tipo que las matrices de entrada
    cv::Mat absA = cv::abs(a);
    cv::Mat absB = cv::abs(b);

    // Calcular el mínimo de los valores absolutos
    cv::Mat minAbs;
    cv::min(absA, absB, minAbs);

    // Calcular los signos de las matrices
    cv::Mat signA = sign(a);
    cv::Mat signB = sign(b);

    // Calcular el resultado Min-mod
    cv::Mat result = (signA + signB) / 2.0;

    // Multiplicar elemento a elemento con minAbs
    result = result.mul(minAbs);

    return result;
}

/// Función lambda para calcular gradientes mediante derivadas centrales
auto calculateGradientByCentralDiff(const cv::Mat& input, cv::Mat& gradX, cv::Mat& gradY)
{
    cv::Mat gradXfi, gradXbi, gradYfj, gradYbj;

    calculateForwardBackward(input, gradXfi, gradXbi, true);
    calculateForwardBackward(input, gradYfj, gradYbj, false);

    gradX = minmodM(gradXfi, gradXbi);
    gradY = minmodM(gradYfj, gradYbj);
};





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                             Cálculo variacional ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Gradiente
cv::Mat nabla(const cv::Mat& I) {
    int h = I.rows;
    int w = I.cols;

    cv::Mat Gx(h, w, CV_32F, cv::Scalar(0));
    cv::Mat Gy(h, w, CV_32F, cv::Scalar(0));

    Gx(cv::Range::all(), cv::Range(0, w - 1)) =
        I(cv::Range::all(), cv::Range(1, w)) - I(cv::Range::all(), cv::Range(0, w - 1));

    Gy(cv::Range(0, h - 1), cv::Range::all()) =
        I(cv::Range(1, h), cv::Range::all()) - I(cv::Range(0, h - 1), cv::Range::all());

    std::vector<cv::Mat> gradients = {Gx, Gy};
    cv::Mat G;
    cv::merge(gradients, G);

    return G;
}

/// Gradiente conjugado (transpuesto)
cv::Mat nablaT(const cv::Mat& G) {
    int h = G.rows;
    int w = G.cols;

    cv::Mat I(h, w, CV_32F, cv::Scalar(0));

    std::vector<cv::Mat> channels(2);
    cv::split(G, channels);

    cv::Mat Gx = channels[0];
    cv::Mat Gy = channels[1];

    I(cv::Range::all(), cv::Range(0, w - 1)) -= Gx(cv::Range::all(), cv::Range(0, w - 1));
    I(cv::Range::all(), cv::Range(1, w)) += Gx(cv::Range::all(), cv::Range(0, w - 1));

    I(cv::Range(0, h - 1), cv::Range::all()) -= Gy(cv::Range(0, h - 1), cv::Range::all());
    I(cv::Range(1, h), cv::Range::all()) += Gy(cv::Range(0, h - 1), cv::Range::all());

    return I;
}

/// Función para calcular la norma L2
cv::Mat normL2(const cv::Mat& x)
{
    cv::Mat norm = cv::Mat::zeros(x.size[0], x.size[1], CV_32F);
    for (int i = 0; i < x.size[0]; i++)
    {
        for (int j = 0; j < x.size[1]; j++)
        {
            norm.at<float>(i, j) = sqrt(x.at<cv::Vec2f>(i, j)[0] * x.at<cv::Vec2f>(i, j)[0] +
                                        x.at<cv::Vec2f>(i, j)[1] * x.at<cv::Vec2f>(i, j)[1]);
        }
    }

    return norm;
}

/// Función para calcular la norma L1
cv::Mat normL1(const cv::Mat& x)
{
    cv::Mat norm = cv::Mat::zeros(x.size[0], x.size[1], CV_32F);
    for (int i = 0; i < x.size[0]; i++)
    {
        for (int j = 0; j < x.size[1]; j++)
        {
            norm.at<float>(i, j) = abs(x.at<cv::Vec2f>(i, j)[0]) +
                                   abs(x.at<cv::Vec2f>(i, j)[1]);
        }
    }
    return norm;
}

/// Función para calcular la norma L∞
cv::Mat normLInf(const cv::Mat& x)
{
    cv::Mat norm = cv::Mat::zeros(x.size[0], x.size[1], CV_32F);
    for (int i = 0; i < x.size[0]; i++)
    {
        for (int j = 0; j < x.size[1]; j++)
        {
            norm.at<float>(i, j) = std::max(abs(x.at<cv::Vec2f>(i, j)[0]),
                                             abs(x.at<cv::Vec2f>(i, j)[1]));
        }
    }
    return norm;
}

/// Función para calcular la norma Frobenius de una matriz
cv::Mat normFrobenius(const cv::Mat& x)
{
    cv::Mat norm = cv::Mat::zeros(x.size(), CV_32F);
    for (int i = 0; i < x.rows; i++)
    {
        for (int j = 0; j < x.cols; j++)
        {
            norm.at<float>(i, j) =
                x.at<cv::Vec2f>(i, j)[0] * x.at<cv::Vec2f>(i, j)[0] +
                x.at<cv::Vec2f>(i, j)[1] * x.at<cv::Vec2f>(i, j)[1];
        }
    }
    return norm;
}

/// Función para calcular la norma Lp
cv::Mat normLp(const cv::Mat& x, int p)
{
    cv::Mat norm = cv::Mat::zeros(x.size(), CV_32F);

    for (int i = 0; i < x.rows; i++)
    {
        for (int j = 0; j < x.cols; j++)
        {
            norm.at<float>(i, j) = std::pow(std::pow(std::fabs(x.at<cv::Vec2f>(i, j)[0]), p) +
                                   std::pow(std::fabs(x.at<cv::Vec2f>(i, j)[1]), p), 1.0f / p);
        }
    }

    return norm;
}

/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxLinfUnitBall(cv::Mat &X)
{
    for (int y = 0; y < X.rows; ++y)
    {
        float *p_x = X.ptr<float>(y);

        for (int x = 0; x < X.cols; ++x, ++p_x)
        {
            float normX = std::fabs(*p_x);
            normX = MIN(1.0, normX);

            *p_x /= normX;
        }
    }
}

/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxL2UnitBall(cv::Mat &X)
{
    if (!X.data)
    {
        return;
    }

    for (int y = 0; y < X.rows; ++y)
    {
        float *p_x = X.ptr<float>(y);

        for (int x = 0; x < X.cols; ++x, ++p_x)
        {
            if (std::fabs(*p_x) > 1.0)
            {
                *p_x = (*p_x > 0.0 ? 1.0 : -1.0);
            }
        }
    }
}

/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxL2Ball(cv::Mat &X, cv::Mat const &center, float radius = 1.0)
{
    if (!X.data)
    {
        return;
    }

    // Shift the center
    if (center.data)
    {
        X -= center;
    }

    // Normalize the coordinates
    X /= radius;

    // Apply projection
    ProxL2UnitBall(X);

    // Un-normalize
    X *= radius;

    // Shift back
    if (center.data)
    {
        X += center;
    }
}

/**
 * Solves the proximal operator associated with the L2 dataterm, i.e. solves the problem:
 * 		argmin { |Y-X|^2/(2*tau) + lambda*|Y-dataTerm|^2 }
 * X is modified by the function, i.e. contains the solution.
 */
/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxL2(cv::Mat &X, cv::Mat const &dataTerm, float lambda, float tau)
{
    float lambdaTau = lambda*tau;

    X += (lambdaTau * dataTerm);
    X /= (1.0 + lambdaTau);
}

/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxLinfBall(cv::Mat &X, cv::Mat const &center, float radius = 1.0f)
{
    if (!X.data)
    {
        return;
    }

    // Shift the center
    if (center.data)
    {
        X -= center;
    }

    // Normalize the coordinates
    X /= radius;

    // Apply projection
    ProxLinfUnitBall(X);

    // Un-normalize
    X *= radius;

    // Shift back
    if (center.data)
    {
        X += center;
    }
}

/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxLinfUnitBall(cv::Mat &X1, cv::Mat &X2)
{
    #define SQUARED_NORM(X,Y) ((X)*(X)+(Y)*(Y))

    int cols = MIN(X1.cols, X2.cols);
    int rows = MIN(X1.rows, X2.rows);

    for (int y = 0; y < rows; ++y)
    {
        float *p_x1 = X1.ptr<float>(y);
        float *p_x2 = X2.ptr<float>(y);

        for (int x = 0; x < cols; ++x, ++p_x1, ++p_x2)
        {
            float normX = std::sqrt(SQUARED_NORM(*p_x1, *p_x2));
            normX = MAX(1.0, normX);

            *p_x1 /= normX;
            *p_x2 /= normX;
        }
    }
}

/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxLinfBall(cv::Mat &X1, cv::Mat &X2, cv::Mat const &C1 = cv::Mat(), cv::Mat const &C2 = cv::Mat(), float radius = 1.0f)
{
    if (!X1.data)
    {
        return;
    }

    // Shift the center
    if (C1.data)
    {
        X1 -= C1;
    }

    if (C2.data)
    {
        X2 -= C2;
    }

    // Normalize the coordinates
    X1 /= radius;
    X2 /= radius;

    // Apply projection
    ProxLinfUnitBall(X1, X2);

    // Un-normalize
    X1 *= radius;
    X2 *= radius;

    // Shift back
    if (C1.data)
    {
        X1 += C1;
    }
    if (C2.data)
    {
        X2 += C2;
    }
}

/**
 * Solves the proximal operator associated with the L2 inpainting dataterm, i.e. solves the problem:
 * 		argmin { |Y-X|^2/(2*tau) + lambda*|A(Y)-dataTerm|^2 }
 * X is modified by the function, i.e. contains the solution.
 */
/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxL2Inpainting(cv::Mat &X, cv::Mat const &dataTerm, cv::Mat const &mask)
{
    for (int y = 0; y < X.rows; ++y)
    {
        float *p_x = X.ptr<float>(y);
        float const *p_mask = mask.ptr<float>(y);
        float const *p_data = dataTerm.ptr<float>(y);

        for (int x = 0; x < X.cols; ++x, ++p_x, ++p_mask, ++p_data)
        {
            if (*p_mask)
            {
                *p_x = *p_data;
            }
        }
    }
}

/**
 * Computes the proximal operator of the indicator function of the set [xmin,xmax],
 * i.e. thresholds x to be in this set.
 */
/// @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/lib/tv/primaldual.cpp
void ProxInterval(cv::Mat &X, float xmin, float xmax)
{
    for (int y = 0; y < X.rows; ++y)
    {
        float *p_x = X.ptr<float>(y);

        for (int x = 0; x < X.cols; ++x, ++p_x)
        {
            *p_x = MIN(MAX(xmin, *p_x), xmax);
        }
    }
}





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                       Deburring ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat applyGaussianBlur(const cv::Mat& inputImage, int kernelSize = 5, double sigma = 1.0) {
    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), sigma);
    return outputImage;
}

cv::Mat applyMotionBlur(const cv::Mat& inputImage, int kernelSize = 5, double angle = 0.0) {
    cv::Mat outputImage;
    cv::Mat motionKernel = cv::Mat::zeros(kernelSize, kernelSize, CV_64F);

    double radians = angle * CV_PI / 180.0;
    int centerX = kernelSize / 2;
    int centerY = kernelSize / 2;

    for (int i = 0; i < kernelSize; ++i) {
        int x = static_cast<int>(centerX + (i - centerY) * sin(radians));
        int y = static_cast<int>(centerY - (i - centerY) * cos(radians));

        if (x >= 0 && x < kernelSize && y >= 0 && y < kernelSize) {
            motionKernel.at<double>(y, x) = 1.0;
        }
    }

    motionKernel /= cv::sum(motionKernel)[0];

    cv::filter2D(inputImage, outputImage, CV_64F, motionKernel);
    return outputImage;
}

cv::Mat applyBoxBlur(const cv::Mat& inputImage, int kernelSize = 5) {
    cv::Mat outputImage;
    cv::blur(inputImage, outputImage, cv::Size(kernelSize, kernelSize));
    return outputImage;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                       Denoising ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat addGaussianNoise(const cv::Mat& image, float sigma)
{
    cv::Mat noise(image.size(), image.type());

    cv::randn(noise, 0, sigma);

    return image + noise;
}

cv::Mat addImpulseNoise(const cv::Mat& image, float saltProbability, float pepperProbability)
{
    cv::Mat noisy_image = image.clone();

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < noisy_image.rows; i++)
    {
        for (int j = 0; j < noisy_image.cols; j++)
        {
            float randomValue = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            if (randomValue < saltProbability)
            {
                if (noisy_image.channels() == 1)
                    noisy_image.at<float>(i, j) = 255.0f;
                else
                    noisy_image.at<cv::Vec3f>(i, j) = cv::Vec3f(255.0f, 255.0f, 255.0f);
            }
            else if (randomValue < (saltProbability + pepperProbability))
            {
                if (noisy_image.channels() == 1)
                    noisy_image.at<float>(i, j) = 0.0f;
                else
                    noisy_image.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }
        }
    }

    return noisy_image;
}

/// Función para crear una imagen con manchas
cv::Mat make_spotty(const cv::Mat& img, int r = 3, int n = 1000)
{
    cv::Mat spotty_img = img.clone();
    int h = img.rows;
    int w = img.cols;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, std::min(h - r, w - r));

    for (int i = 0; i < n; i++)
    {
        int x = distribution(generator);
        int y = distribution(generator);
        spotty_img(cv::Rect(x, y, r, r)) = cv::Scalar(round((float)rand() / (float)RAND_MAX));
    }
    return spotty_img;
}

/// Función para crear una imagen ruidosa
cv::Mat make_noisy(const cv::Mat &img) {
    if (img.empty()) {
        throw std::runtime_error("La imagen de entrada está vacía.");
    }

    cv::Mat noisy_img = img.clone();

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            float noise = distribution(generator);
            noisy_img.at<float>(y, x) += noise;
        }
    }

    std::uniform_real_distribution<float> uniform_distribution(0.0f, 1.0f);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (uniform_distribution(generator) < 0.2f) {
                if (x >= 160) {
                    noisy_img.at<float>(y, x) = uniform_distribution(generator);
                }
            }
        }
    }

    return noisy_img;
}

void createRandomInpaintingMask(float occlusionRatio, cv::Mat &mask)
{
    mask.setTo(cv::Scalar(1));

    occlusionRatio = std::max(std::min(occlusionRatio, 1.0f), 0.0f);

    if (mask.type() == CV_32FC1) {
        cv::randu(mask, cv::Scalar(0), cv::Scalar(1));
    } else if (mask.type() == CV_64FC1) {
        cv::randu(mask, cv::Scalar(0.0), cv::Scalar(1.0));
    }

    if (mask.type() == CV_32FC1) {
        for (int y = 0; y < mask.size().height; ++y) {
            float *p_mask = mask.ptr<float>(y);
            for (int x = 0; x < mask.size().width; ++x) {
                if (*p_mask < occlusionRatio) {
                    *p_mask = 0.0f;
                } else {
                    *p_mask = 1.0f;
                }
                ++p_mask;
            }
        }
    } else if (mask.type() == CV_64FC1) {
        for (int y = 0; y < mask.size().height; ++y) {
            double *p_mask = mask.ptr<double>(y);
            for (int x = 0; x < mask.size().width; ++x) {
                if (*p_mask < occlusionRatio) {
                    *p_mask = 0.0;
                } else {
                    *p_mask = 1.0;
                }
                ++p_mask;
            }
        }
    }
}





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                         Fourier ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat makeMultiChannel(cv::Mat& src, int newChannels)
{
    cv::Mat multiChanel(src.size(), CV_MAKETYPE(src.depth(), newChannels));
    std::vector<cv::Mat> channels(newChannels);
    for (int i = 0; i < newChannels; ++i)
        channels[i] = src;
    cv::merge(channels, multiChanel);
    return multiChanel;
}

















///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                   Visualización ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// MATLAB's imshowpair()
cv::Mat imshowpair(std::string title, cv::Mat const& leftImage, cv::Mat const& rightImage)
{
    int totalWidth = 2 * MAX(leftImage.cols, rightImage.cols);
    int totalHeight = MAX(leftImage.rows, rightImage.rows);

    cv::Mat composite;

    composite.create(cv::Size(totalWidth, totalHeight), leftImage.type());
    composite.setTo(cv::Scalar::all(0));

    cv::Mat targetROI;

    int xoffset = -(leftImage.cols - totalWidth/2) / 2;
    int yoffset = -(leftImage.rows - totalHeight) / 2;

    targetROI = composite(cv::Rect(xoffset, yoffset, leftImage.cols, leftImage.rows));
    leftImage.copyTo(targetROI);

    xoffset = (totalWidth/2) - (rightImage.cols - totalWidth/2)/2;
    yoffset = -(rightImage.rows - totalHeight)/2;

    targetROI = composite(cv::Rect(xoffset, yoffset, rightImage.cols, rightImage.rows));
    rightImage.copyTo(targetROI);

    cv::imshow(title, composite);

    return composite;
}

void muestraImagenOpenCV(const cv::Mat img, std::string title, bool destroyAfter = true)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::resizeWindow(title, 800, 600);
    cv::imshow(title, img);
    cv::waitKey(0);

    if (destroyAfter)
        cv::destroyWindow(title);
}

void MyTimeOutput(const std::string& str, const std::chrono::high_resolution_clock::time_point& start_time, const std::chrono::high_resolution_clock::time_point& end_time)
{
    std::cout << str << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0 << " ms" << std::endl;
    return;
}

void plotCostGraph(const std::vector<double>& costs)
{
    int graphWidth = 400;
    int graphHeight = 300;
    cv::Mat graph(graphHeight, graphWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // Encontrar el valor máximo de coste para normalizar
    double maxCost = *std::max_element(costs.begin(), costs.end());

    // Dibujar el gráfico
    for (size_t i = 1; i < costs.size(); ++i)
    {
        cv::line(graph,
                 cv::Point((i - 1) * graphWidth / costs.size(), graphHeight - costs[i - 1] * graphHeight / maxCost),
                 cv::Point(i * graphWidth / costs.size(), graphHeight - costs[i] * graphHeight / maxCost),
                 cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Cost Graph", graph);
}

// Función para graficar una señal 1D usando gnuplot
void plotSignal1DGnuplot(const cv::Mat& g, const std::string& plotTitle) {
    // Verificar que la imagen sea válida y en escala de grises
    if (g.empty() || g.channels() != 1) {
        std::cerr << "La imagen no es válida o no está en escala de grises!" << std::endl;
        return;
    }

    // Obtener la fila central de la imagen
    int plotRow = g.rows / 2;
    cv::Mat rowPlot = g.row(plotRow).clone();

    // Abrir el pipe a gnuplot
    FILE* gnuplotPipe = popen("gnuplot -p", "w");
    if (!gnuplotPipe) {
        std::cerr << "Error al abrir el pipe a gnuplot!" << std::endl;
        return;
    }

    // Enviar los datos de la fila central a gnuplot con el título personalizado
    fprintf(gnuplotPipe, "plot '-' with lines title '%s'\n", plotTitle.c_str());
    for (int i = 0; i < rowPlot.cols; ++i) {
        fprintf(gnuplotPipe, "%d %d\n", i, static_cast<int>(rowPlot.at<uchar>(0, i)));
    }
    fprintf(gnuplotPipe, "e\n");  // Indicar el fin de los datos para gnuplot

    // Cerrar el pipe de gnuplot
    fclose(gnuplotPipe);
}















///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                              3D ///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Leer un .txt que contenga las coordenadas de una nube de puntos (puntos == filas, dimensiones xyz == columnas)
/// y guardar la nube en un cv::Mat CV_64FC3
cv::Mat readCoordinatesFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        return cv::Mat();

    std::vector<cv::Vec3d> points;
    std::string line;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        double x, y, z;
        if (iss >> x >> y >> z)
            points.emplace_back(x, y, z);
    }


    cv::Mat mat(1, points.size(), CV_64FC3);
    cv::Vec3d* matPtr = mat.ptr<cv::Vec3d>(0);
    for (size_t i = 0; i < points.size(); ++i)
        matPtr[i] = points[i];

    return mat;
}

/// Guardar la nube en un cv::Mat CV_64FC3 en un .txt las coordenadas de una nube de
/// puntos (puntos == filas, dimensiones xyz == columnas)
bool saveCoordinatesToFile(const cv::Mat& mat, const std::string& filename)
{
    if (mat.empty() || mat.channels() != 3 || mat.type() != CV_64FC3)
        return false;

    std::ofstream file(filename);
    if (!file.is_open())
        return false;

    const double* ptr = mat.ptr<double>(0);
    for (int i = 0; i < mat.cols; ++i)
        file << ptr[i * 3] << " " << ptr[i * 3 + 1] << " " << ptr[i * 3 + 2] << std::endl;

    file.close();

    return true;
}

}
