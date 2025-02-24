#include "HarmonicImageInpainting.h"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#include <QDebug>

HarmonicImageInpainting::HarmonicImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void HarmonicImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("nIters"),
                                1000,
                                QString("Iterations number"),
                                QString("Iterations number"),
                                true, 1, 9999);
    mParameterSet->addParameter(QString("dt"),
                                0.15,
                                QString("Time step"),
                                QString("Time step"),
                                true, 0.0001, 100.0);
    mParameterSet->addParameter(QString("alpha"),
                                1.5,
                                QString("alpha"),
                                QString("alpha"),
                                true, 0.0, 100.0);
    mParameterSet->addParameter(QString("lambda0"),
                                1.0,
                                QString("Diffusion coefficient"),
                                QString("Diffusion coefficient"),
                                true, 0.0, 100.0);

    mParameterSet->setName(QString("Heat (Harmonic) Equation Image Inpainting"));
    mParameterSet->setLabel(QString("Heat (Harmonic) Equation Image Inpainting"));
    mParameterSet->setIntroduction(QString("Parameters settings"));
}

void HarmonicImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getOriginalImage();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }

    cv::Mat mask = mDataManager->getMask();
    if (mask.empty())
    {
        qWarning() << "The mask is empty";
        return;
    }

    const bool useMatrixForm = false;
    int nIters;
    double dt, alpha, lambda0;
    if (!mParameterSet->getValue(QString("nIters"), nIters))
    {
        qWarning() << "Did not find 'nIters'";
        return;
    }
    if (!mParameterSet->getValue(QString("dt"), dt))
    {
        qWarning() << "Did not find 'dt'";
        return;
    }
    if (!mParameterSet->getValue(QString("alpha"), alpha))
    {
        qWarning() << "Did not find 'alpha'";
        return;
    }
    if (!mParameterSet->getValue(QString("lambda0"), lambda0))
    {
        qWarning() << "Did not find 'lambda0'";
        return;
    }

    cv::Mat lambda = cv::Mat::zeros(image.size(), CV_64FC1);
    lambda.setTo(lambda0, mask == 255);

    cv::Mat inpainted;

    if (image.channels() == 4)
    {
        if (image.depth() == CV_64F)
            image.convertTo(image, CV_32F);
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }

    if (image.channels() == 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        for (int i = 0; i < (int)channels.size(); ++i)
        {
            emit sendOtherMessage("Processing channel " + QString::number(i) + "...");
            channels[i] = universalConvertTo(channels[i], CV_64FC1);
            channels[i].setTo(0.0, ~mask);
            cv::Mat inpaintedChannel;
            heatEquationHarmonicInpainting(channels[i], inpaintedChannel, lambda, nIters, dt, alpha, useMatrixForm);
            channels[i] = inpaintedChannel;
        }
        emit sendOtherMessage("");

        cv::merge(channels, inpainted);
    }
    else
    {
        emit sendOtherMessage("Processing image...");
        cv::Mat g = universalConvertTo(image, CV_64FC1);
        g.setTo(0.0, ~mask);
        heatEquationHarmonicInpainting(g, inpainted, lambda, nIters, dt, alpha, useMatrixForm);
        emit sendOtherMessage("");
    }

    mDataManager->setImage(image);
    mDataManager->setInpaintedImage(inpainted);

    emit sendImageProcess(inpainted);
}

//void HarmonicImageInpainting::heatEquationHarmonicInpainting(const cv::Mat &g, cv::Mat &u, const cv::Mat &lambda, int nIters, double dt, double alpha, bool useMatrixForm)
//{
//    int height = g.rows;
//    int width = g.cols;

//    u = g.clone();
//    if (!useMatrixForm)
//    {
//        const double* gPtr = g.ptr<double>();

//        for (int t = 0; t < nIters; t++)
//        {
//            cv::Mat u_prev = u.clone();

//            double* uPtr = u.ptr<double>();
//            double* u_prevPtr = u_prev.ptr<double>();

//            for (int r = 0; r < height; ++r)
//            {
//                for (int c = 0; c < width; ++c)
//                {
//                    double d2x{0.0}, d2y{0.0};

//                    const double u_ij = uPtr[r * width + c];
//                    const double u_prev_ij = u_prevPtr[r * width + c];

//                    /// Diferencias centrales y gradiente constante = 0.0 en el contorno para que no funcionen como fuentes ni sumideros de calor (divergencia es cero == condiciones de contorno de Neumann == sistema aislado)
//                    if (c != 0 && c != width - 1)
//                        d2x = uPtr[r * width + (c + 1)] - 2 * u_ij + uPtr[r * width + (c - 1)];

//                    if (r != 0 && r != height - 1)
//                        d2y = uPtr[(r + 1) * width + c] - 2 * u_ij + uPtr[(r - 1) * width + c];

//                    uPtr[r * width + c] += dt * (alpha * (d2x + d2y) + lambda.at<double>(r, c) * (gPtr[r * width + c] - u_prev_ij));
//                }
//            }

//            emit sendImageProcess(u);
//        }
//    }
//    else
//    {
//        cv::Mat laplacian;
//        for (int t = 0; t < nIters; ++t)
//        {
//            cv::Laplacian(u, laplacian, CV_64F, 1, 1.0, 0.0, cv::BORDER_REFLECT);

//            u += dt * (alpha * laplacian + lambda.mul(g - u));
////            u.setTo(1.0, u > 1.0);
////            u.setTo(0.0, u < 0.0);

//            emit sendImageProcess(u);
//        }
//    }

//    // Finalmente hacer clamp de valores remanentes que pudieran haber quedado fuera de rango
//    u.setTo(1.0, u > 1.0);
//    u.setTo(0.0, u < 0.0);
//}

void HarmonicImageInpainting::heatEquationHarmonicInpainting(const cv::Mat &g, cv::Mat &u, const cv::Mat &lambda, int nIters, double dt, double alpha, bool useMatrixForm)
{
    int height = g.rows;
    int width = g.cols;

    u = g.clone();  // Initialize u with the original image g.

    if (!useMatrixForm)
    {
        const double* gPtr = g.ptr<double>();

        for (int t = 0; t < nIters; t++)
        {
            cv::Mat u_prev = u.clone();
            double* uPtr = u.ptr<double>();
            double* u_prevPtr = u_prev.ptr<double>();

            for (int r = 0; r < height; ++r)
            {
                for (int c = 0; c < width; ++c)
                {
                    if (lambda.at<double>(r, c) > 0)  // If lambda is non-zero
                        uPtr[r * width + c] = gPtr[r * width + c];  // Enforce the original image value in u
                    else
                    {
                        double d2x{0.0}, d2y{0.0};

                        const double u_ij = uPtr[r * width + c];
                        const double u_prev_ij = u_prevPtr[r * width + c];

                        if (c != 0 && c != width - 1)
                            d2x = uPtr[r * width + (c + 1)] - 2 * u_ij + uPtr[r * width + (c - 1)];

                        if (r != 0 && r != height - 1)
                            d2y = uPtr[(r + 1) * width + c] - 2 * u_ij + uPtr[(r - 1) * width + c];

                        uPtr[r * width + c] += dt * (alpha * (d2x + d2y) + lambda.at<double>(r, c) * (gPtr[r * width + c] - u_prev_ij));
                    }
                }
            }

            emit sendImageProcess(u);
        }
    }
    else
    {
        cv::Mat laplacian;
        for (int t = 0; t < nIters; ++t)
        {
            cv::Laplacian(u, laplacian, CV_64F, 1, 1.0, 0.0, cv::BORDER_REFLECT);

            u += dt * (alpha * laplacian + lambda.mul(g - u));

            // Ensure u matches g where lambda is non-zero
            for (int r = 0; r < height; ++r)
            {
                for (int c = 0; c < width; ++c)
                {
                    if (lambda.at<double>(r, c) > 0)  // If lambda is non-zero
                        u.at<double>(r, c) = g.at<double>(r, c);  // Enforce the original image value
                }
            }

            emit sendImageProcess(u);
        }
    }

    // Finally, clamp the values to make sure they stay within the [0, 1] range
    u.setTo(1.0, u > 1.0);
    u.setTo(0.0, u < 0.0);
}
